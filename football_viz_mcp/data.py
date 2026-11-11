"""User datasets: loading, storage, filtering and column resolution.

Datasets are supplied by the end user (CSV, Excel, JSON, a public URL or a
Google Sheets link) and kept in memory under a short random id so chart tools
can reference them without re-sending the data on every call.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import os
import pickle
import re
import secrets
import socket
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

MAX_ROWS = int(os.getenv("FOOTBALL_VIZ_MCP_MAX_ROWS", "500000"))
MAX_DOWNLOAD_BYTES = int(os.getenv("FOOTBALL_VIZ_MCP_MAX_DOWNLOAD_MB", "50")) * 1024 * 1024
ALLOW_URLS = os.getenv("FOOTBALL_VIZ_MCP_ALLOW_URLS", "true").lower() in ("1", "true", "yes")
ALLOW_PRIVATE_URLS = os.getenv("FOOTBALL_VIZ_MCP_ALLOW_PRIVATE_URLS", "false").lower() in ("1", "true", "yes")


from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError


class ToolError(_MCPToolError):
    """An error whose message is safe and useful to show to the LLM (surfaced as a tool error)."""


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    id: str
    name: str
    df: pd.DataFrame
    source: str
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def meta(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "source": self.source, "created": self.created,
                "rows": int(len(self.df)), "columns": int(len(self.df.columns))}


# Persistent backends ----------------------------------------------------------
# Datasets always live in memory while in use. A backend keeps a copy so they survive restarts
# (and, on Cloudflare Containers, the container going to sleep). Keys look like
# "datasets/ds_xxx.pkl", "datasets/ds_xxx.json" and "charts/<id>.png".


class LocalBackend:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, key: str) -> str:
        if ".." in key or key.startswith("/"):
            raise ValueError("bad key")
        return os.path.join(self.root, key)

    def save(self, key: str, data: bytes) -> None:
        path = self._p(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)

    def load(self, key: str) -> bytes | None:
        try:
            with open(self._p(key), "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def delete(self, key: str) -> None:
        try:
            os.remove(self._p(key))
        except FileNotFoundError:
            pass

    def list(self, prefix: str) -> list[str]:
        d = self._p(prefix)
        if not os.path.isdir(d):
            return []
        return [prefix + f for f in os.listdir(d) if not f.endswith(".tmp")]


class S3Backend:
    """Cloudflare R2 (or any S3-compatible storage) through boto3."""

    def __init__(self, bucket: str, endpoint: str, key_id: str | None, secret: str | None,
                 prefix: str = "") -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self.client = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id=key_id, aws_secret_access_key=secret,
            region_name=os.getenv("S3_REGION", "auto"),
            config=Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=5, read_timeout=30))

    def save(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data)

    def load(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self.prefix + key)

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket,
                                                                          Prefix=self.prefix + prefix):
            keys += [o["Key"][len(self.prefix):] for o in page.get("Contents", [])]
        return keys


def make_backend() -> LocalBackend | S3Backend | None:
    bucket = os.getenv("R2_BUCKET_NAME") or os.getenv("S3_BUCKET")
    if bucket:
        endpoint = os.getenv("S3_ENDPOINT_URL")
        if not endpoint:
            account = os.getenv("R2_ACCOUNT_ID")
            if not account:
                raise RuntimeError("Set R2_ACCOUNT_ID (or S3_ENDPOINT_URL) together with R2_BUCKET_NAME.")
            endpoint = f"https://{account}.r2.cloudflarestorage.com"
        return S3Backend(bucket, endpoint,
                         os.getenv("R2_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID"),
                         os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"),
                         prefix=os.getenv("S3_PREFIX", "football-viz-mcp"))
    if os.getenv("FOOTBALL_VIZ_MCP_DATA_DIR"):
        return LocalBackend(os.environ["FOOTBALL_VIZ_MCP_DATA_DIR"])
    return None


_ID_RE = re.compile(r"ds_[a-f0-9]{10}")


class DataStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Dataset] = {}
        self.ttl = float(os.getenv("FOOTBALL_VIZ_MCP_DATASET_TTL_HOURS", "72")) * 3600
        self.max_items = int(os.getenv("FOOTBALL_VIZ_MCP_MAX_DATASETS", "200"))
        self.backend = make_backend()

    # persistence ------------------------------------------------------------
    def _persist(self, ds: Dataset) -> None:
        if self.backend:
            self.backend.save(f"datasets/{ds.id}.pkl", pickle.dumps(ds.df, protocol=pickle.HIGHEST_PROTOCOL))
            self.backend.save(f"datasets/{ds.id}.json", json.dumps(ds.meta()).encode())

    def _restore(self, ds_id: str) -> Dataset | None:
        """Load a dataset that is not in memory (e.g. after a restart) from the backend."""
        if not self.backend or not _ID_RE.fullmatch(ds_id):
            return None
        raw_meta = self.backend.load(f"datasets/{ds_id}.json")
        raw_df = self.backend.load(f"datasets/{ds_id}.pkl") if raw_meta else None
        if not raw_meta or not raw_df:
            return None
        meta = json.loads(raw_meta)
        if time.time() - float(meta.get("created", 0)) > self.ttl:
            self._unpersist(ds_id)
            return None
        # Only our own server writes to this private bucket/folder.
        df = pickle.loads(raw_df)  # noqa: S301
        return Dataset(id=ds_id, name=meta.get("name", ds_id), df=df, source=meta.get("source", ""),
                       created=float(meta.get("created", time.time())))

    def _stored_meta(self) -> list[dict[str, Any]]:
        if not self.backend:
            return []
        out = []
        for key in self.backend.list("datasets/"):
            if key.endswith(".json"):
                raw = self.backend.load(key)
                if raw:
                    try:
                        out.append(json.loads(raw))
                    except ValueError:
                        continue
        return out

    def _unpersist(self, ds_id: str) -> None:
        if self.backend:
            for ext in ("pkl", "json"):
                try:
                    self.backend.delete(f"datasets/{ds_id}.{ext}")
                except Exception:  # noqa: BLE001
                    pass

    # api --------------------------------------------------------------------
    def _evict(self) -> None:
        """Free memory. Copies in the backend are kept (they expire by TTL)."""
        now = time.time()
        for ds_id in [k for k, v in self._items.items() if now - v.last_used > self.ttl]:
            self._items.pop(ds_id, None)
            self._unpersist(ds_id)
        while len(self._items) >= self.max_items:
            oldest = min(self._items.values(), key=lambda d: d.last_used)
            self._items.pop(oldest.id, None)

    def put(self, df: pd.DataFrame, name: str, source: str) -> Dataset:
        with self._lock:
            self._evict()
            ds = Dataset(id="ds_" + secrets.token_hex(5), name=name or "dataset", df=df, source=source)
            self._items[ds.id] = ds
        try:
            self._persist(ds)
        except Exception as exc:  # noqa: BLE001 - keep working in memory if storage fails
            import logging
            logging.getLogger(__name__).warning("Could not persist dataset %s: %s", ds.id, exc)
        return ds

    def get(self, ds_id: str) -> Dataset:
        key = ds_id.strip()
        with self._lock:
            ds = self._items.get(key)
            if ds is None:
                matches = [d for d in self._items.values() if d.name == key]
                if len(matches) == 1:
                    ds = matches[0]
        if ds is None:
            if not _ID_RE.fullmatch(key) and self.backend:
                # lookup by (unique) name among stored datasets
                metas = [m for m in self._stored_meta() if m.get("name") == key]
                if len(metas) == 1:
                    key = metas[0]["id"]
            ds = self._restore(key)
            if ds is not None:
                with self._lock:
                    self._evict()
                    self._items[ds.id] = ds
        if ds is None:
            raise ToolError(
                f"Dataset '{ds_id}' not found (it may have expired). "
                "Use list_datasets to see available ids, or load the data again with load_dataset."
            )
        ds.last_used = time.time()
        return ds

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            self._evict()
            metas = {d.id: d.meta() for d in self._items.values()}
        for m in self._stored_meta():
            if time.time() - float(m.get("created", 0)) <= self.ttl:
                metas.setdefault(m["id"], m)
        return sorted(metas.values(), key=lambda m: m.get("created", 0))

    def delete(self, ds_id: str) -> bool:
        with self._lock:
            removed = self._items.pop(ds_id, None) is not None
        if self.backend and self.backend.load(f"datasets/{ds_id}.json") is not None:
            removed = True
        self._unpersist(ds_id)
        return removed

    # charts -----------------------------------------------------------------
    def save_chart(self, cid: str, png: bytes) -> None:
        if self.backend:
            try:
                self.backend.save(f"charts/{cid}.png", png)
            except Exception:  # noqa: BLE001
                pass

    def load_chart(self, cid: str) -> bytes | None:
        if self.backend and re.fullmatch(r"[A-Za-z0-9_-]{6,40}", cid):
            try:
                return self.backend.load(f"charts/{cid}.png")
            except Exception:  # noqa: BLE001
                return None
        return None


STORE = DataStore()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _check_url(url: str) -> str:
    if not ALLOW_URLS:
        raise ToolError("Loading data from URLs is disabled on this server.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("Only http(s) URLs are supported.")
    if not ALLOW_PRIVATE_URLS:
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise ToolError(f"Could not resolve host '{parsed.hostname}'.") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ToolError("URLs pointing to private or local network addresses are not allowed.")
    return url


def _google_sheets_to_csv(url: str) -> str:
    m = re.match(r"https://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)", url)
    if not m or "export?format=" in url or "/pub?" in url:
        return url
    gid = re.search(r"[#&?]gid=(\d+)", url)
    out = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    return out + (f"&gid={gid.group(1)}" if gid else "")


class _NoPrivateRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url(url: str) -> tuple[bytes, str]:
    url = _check_url(_google_sheets_to_csv(url.strip()))
    opener = urllib.request.build_opener(_NoPrivateRedirect())
    req = urllib.request.Request(url, headers={"User-Agent": "football-viz-mcp/1.0"})
    try:
        with opener.open(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Could not download the URL: {exc}") from exc
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ToolError(f"The file is larger than the {MAX_DOWNLOAD_BYTES // 1024 // 1024} MB limit.")
    return data, ctype


def _guess_format(name: str, ctype: str, data: bytes) -> str:
    lname = name.lower().split("?")[0]
    for ext, fmt in ((".csv", "csv"), (".tsv", "csv"), (".txt", "csv"), (".json", "json"),
                     (".jsonl", "jsonl"), (".ndjson", "jsonl"), (".xlsx", "excel"), (".xls", "excel"),
                     (".parquet", "parquet")):
        if lname.endswith(ext):
            return fmt
    if "json" in ctype:
        return "json"
    if "spreadsheetml" in ctype or "excel" in ctype or data[:2] == b"PK":
        return "excel"
    head = data[:200].lstrip()
    if head[:1] in (b"[", b"{"):
        return "json"
    return "csv"


def _read_csv_text(text: str, sep: str | None = None, decimal: str | None = None) -> pd.DataFrame:
    text = text.lstrip("﻿")
    if sep is None:
        first = text.split("\n", 1)[0]
        counts = {s: first.count(s) for s in [",", ";", "\t", "|"]}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
    if decimal is None:
        decimal = "," if sep == ";" else "."
    return pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, low_memory=False)


def _json_to_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, dict):
        # common wrappers: {"data": [...]}, {"events": [...]}, column-oriented dicts
        list_keys = [k for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(list_keys) == 1:
            obj = obj[list_keys[0]]
        elif obj and all(isinstance(v, list) for v in obj.values()):
            return pd.DataFrame(obj)
        else:
            obj = [obj]
    if not isinstance(obj, list):
        raise ToolError("JSON data must be a list of records (objects) or an object containing one.")
    return pd.json_normalize(obj, sep=".")


def parse_bytes(data: bytes, fmt: str, sheet: str | int | None = None, sep: str | None = None,
                decimal: str | None = None) -> pd.DataFrame:
    try:
        if fmt == "csv":
            for enc in ("utf-8", "latin-1"):
                try:
                    return _read_csv_text(data.decode(enc), sep, decimal)
                except UnicodeDecodeError:
                    continue
        if fmt == "json":
            return _json_to_df(json.loads(data.decode("utf-8-sig")))
        if fmt == "jsonl":
            rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
            return _json_to_df(rows)
        if fmt == "excel":
            return pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet is not None else 0)
        if fmt == "parquet":
            return pd.read_parquet(io.BytesIO(data))
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Could not parse the data as {fmt}: {exc}") from exc
    raise ToolError(f"Unsupported format '{fmt}'.")


def _expand_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn columns holding [x, y] / [x, y, z] lists (StatsBomb style) into numeric columns."""
    new_cols = {}
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        sample = s.dropna().head(50)
        if sample.empty or not all(isinstance(v, (list, tuple)) for v in sample):
            continue
        lengths = sample.map(len)
        if lengths.min() < 2 or lengths.max() > 3:
            continue
        for i, suffix in enumerate(["x", "y", "z"][: lengths.max()]):
            new_cols[f"{col}_{suffix}"] = s.map(
                lambda v, i=i: v[i] if isinstance(v, (list, tuple)) and len(v) > i else np.nan
            )
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that are really numbers (incl. '12,5' decimals)."""
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        sample = s.dropna().head(200)
        if sample.empty or not all(isinstance(v, str) for v in sample):
            continue
        cleaned = sample.str.strip().str.replace("%", "", regex=False)
        conv = pd.to_numeric(cleaned, errors="coerce")
        if conv.notna().mean() < 0.95:
            conv = pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")
            if conv.notna().mean() < 0.95:
                continue
            df[col] = pd.to_numeric(s.str.strip().str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
                                    errors="coerce")
        else:
            df[col] = pd.to_numeric(s.str.strip().str.replace("%", "", regex=False), errors="coerce")
    return df


def finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ToolError("The dataset has no rows.")
    if len(df) > MAX_ROWS:
        raise ToolError(f"The dataset has {len(df)} rows; the limit is {MAX_ROWS}.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        seen: dict[str, int] = {}
        cols = []
        for c in df.columns:
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        df.columns = cols
    df = _expand_list_columns(df)
    df = _coerce_numeric(df)
    return df.reset_index(drop=True)


def load_dataframe(*, csv_text: str | None = None, json_data: Any = None, url: str | None = None,
                   file_base64: str | None = None, file_name: str | None = None,
                   file_format: str | None = None, sheet: str | int | None = None,
                   separator: str | None = None, decimal: str | None = None) -> tuple[pd.DataFrame, str]:
    provided = [p for p in (csv_text, json_data, url, file_base64) if p not in (None, "")]
    if len(provided) != 1:
        raise ToolError("Provide exactly one of: csv_text, json_data, url, file_base64.")
    if csv_text:
        try:
            return finalize_df(_read_csv_text(csv_text, separator, decimal)), "csv_text"
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Could not parse csv_text: {exc}") from exc
    if json_data is not None:
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as exc:
                raise ToolError(f"json_data is not valid JSON: {exc}") from exc
        return finalize_df(_json_to_df(json_data)), "json_data"
    if url:
        data, ctype = fetch_url(url)
        fmt = file_format or _guess_format(_google_sheets_to_csv(url), ctype, data)
        return finalize_df(parse_bytes(data, fmt, sheet, separator, decimal)), f"url:{url}"
    try:
        raw = base64.b64decode(file_base64 or "", validate=False)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("file_base64 is not valid base64.") from exc
    fmt = file_format or _guess_format(file_name or "", "", raw)
    return finalize_df(parse_bytes(raw, fmt, sheet, separator, decimal)), f"file:{file_name or fmt}"


# --------------------------------------------------------------------------- #
# Filtering & columns
# --------------------------------------------------------------------------- #


class Filter(BaseModel):
    """A row filter. Several filters are combined with AND."""

    column: str = Field(description="Column name (case-insensitive).")
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "not_contains",
                "startswith", "isnull", "notnull", "between"] = Field(
        default="==", description="Comparison operator. 'between' takes [min, max]; 'in'/'not_in' take a list.")
    value: Any = Field(default=None, description="Value to compare against (not needed for isnull/notnull).")


def resolve_column(df: pd.DataFrame, name: str | None, what: str = "column", required: bool = True) -> str | None:
    if name is None or name == "":
        if required:
            raise ToolError(f"A {what} column is required.")
        return None
    if name in df.columns:
        return name
    lowered = {c.lower(): c for c in df.columns}
    if name.lower() in lowered:
        return lowered[name.lower()]
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    normed = {norm(c): c for c in df.columns}
    if norm(name) in normed:
        return normed[norm(name)]
    close = [c for c in df.columns if norm(name) in norm(c)][:8]
    hint = f" Similar columns: {close}." if close else f" Available columns: {list(df.columns)[:60]}."
    raise ToolError(f"Column '{name}' ({what}) not found.{hint}")


_CANDIDATES: dict[str, list[str]] = {
    "x": ["x", "location_x", "start_x", "x_start", "startx", "location.x", "pos_x", "coord_x", "X"],
    "y": ["y", "location_y", "start_y", "y_start", "starty", "location.y", "pos_y", "coord_y", "Y"],
    "end_x": ["end_x", "endx", "x_end", "pass.end_location_x", "pass_end_location_x", "pass_end_x",
              "carry.end_location_x", "end_location_x", "pass.endlocation.x", "passendx", "x2"],
    "end_y": ["end_y", "endy", "y_end", "pass.end_location_y", "pass_end_location_y", "pass_end_y",
              "carry.end_location_y", "end_location_y", "pass.endlocation.y", "passendy", "y2"],
}


def auto_column(df: pd.DataFrame, given: str | None, role: str, required: bool = True) -> str | None:
    if given:
        return resolve_column(df, given, role)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    normed = {norm(c): c for c in df.columns}
    for cand in _CANDIDATES.get(role, []):
        if norm(cand) in normed:
            return normed[norm(cand)]
    if required:
        raise ToolError(
            f"Could not detect the '{role}' column automatically. Pass it explicitly. "
            f"Available columns: {list(df.columns)[:60]}"
        )
    return None


def apply_filters(df: pd.DataFrame, filters: list[Filter] | None) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for f in filters:
        if isinstance(f, dict):
            f = Filter(**f)
        col = resolve_column(df, f.column, "filter")
        s = df[col]
        v = f.value
        try:
            if f.op == "==":
                m = s.astype(str).str.lower() == str(v).lower() if isinstance(v, str) else s == v
            elif f.op == "!=":
                m = s.astype(str).str.lower() != str(v).lower() if isinstance(v, str) else s != v
            elif f.op in (">", ">=", "<", "<="):
                num = pd.to_numeric(s, errors="coerce")
                m = {">": num > v, ">=": num >= v, "<": num < v, "<=": num <= v}[f.op]
            elif f.op in ("in", "not_in"):
                vals = v if isinstance(v, list) else [v]
                lower = {str(x).lower() for x in vals}
                m = s.astype(str).str.lower().isin(lower) | s.isin(vals)
                if f.op == "not_in":
                    m = ~m
            elif f.op in ("contains", "not_contains"):
                m = s.astype(str).str.contains(str(v), case=False, na=False, regex=False)
                if f.op == "not_contains":
                    m = ~m
            elif f.op == "startswith":
                m = s.astype(str).str.lower().str.startswith(str(v).lower())
            elif f.op == "isnull":
                m = s.isna()
            elif f.op == "notnull":
                m = s.notna()
            elif f.op == "between":
                lo, hi = v
                num = pd.to_numeric(s, errors="coerce")
                m = (num >= lo) & (num <= hi)
            else:  # pragma: no cover
                raise ToolError(f"Unknown operator {f.op}")
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Filter on '{f.column}' failed: {exc}") from exc
        mask &= m.fillna(False).astype(bool)
    return df[mask]


def get_df(dataset_id: str | None, data: list[dict[str, Any]] | None,
           filters: list[Filter] | None = None) -> pd.DataFrame:
    if data:
        df = finalize_df(_json_to_df(data))
    elif dataset_id:
        df = STORE.get(dataset_id).df
    else:
        raise ToolError("Provide a dataset_id (from load_dataset) or inline 'data' records.")
    out = apply_filters(df, filters)
    return out


def numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def suggest_pitch_type(df: pd.DataFrame) -> str | None:
    xc = auto_column(df, None, "x", required=False)
    yc = auto_column(df, None, "y", required=False)
    if not xc or not yc:
        return None
    x, y = numeric(df, xc), numeric(df, yc)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if x.size == 0:
        return None
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    if xmin < -100 or ymin < -100:
        return "tracab (centimetres, origin at centre)"
    if xmin < 0 or ymin < 0:
        return "skillcorner / secondspectrum / impect (metres, origin at centre)"
    if xmax <= 1.01 and ymax <= 1.01:
        return "metricasports"
    if xmax <= 100.5 and ymax <= 100.5 and ymax > 81:
        return "opta or wyscout (0-100); opta has y=0 at the bottom, wyscout at the top"
    if xmax <= 120.5 and ymax <= 80.5 and xmax > 106:
        return "statsbomb"
    if xmax <= 105.5 and ymax <= 68.5:
        return "uefa or custom (metres) — or opta/wyscout if data never reaches the edges"
    if xmax <= 100.5 and ymax <= 100.5:
        return "opta or wyscout (0-100)"
    return "custom (set pitch_length / pitch_width)"


def describe(df: pd.DataFrame, max_unique: int = 25, sample_rows: int = 5) -> dict[str, Any]:
    cols = []
    for c in df.columns:
        s = df[c]
        info: dict[str, Any] = {"name": c, "dtype": str(s.dtype), "non_null": int(s.notna().sum())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if s.notna().any():
                info.update(min=float(np.nanmin(s)), max=float(np.nanmax(s)), mean=round(float(np.nanmean(s)), 4))
        else:
            try:
                nunique = int(s.nunique(dropna=True))
            except TypeError:
                nunique = -1
            info["unique"] = nunique
            if 0 < nunique <= max_unique:
                info["values"] = {str(k): int(v) for k, v in s.value_counts().head(max_unique).items()}
            elif nunique > max_unique:
                info["examples"] = [str(v) for v in s.dropna().unique()[:5]]
        cols.append(info)
    sample = json.loads(df.head(sample_rows).to_json(orient="records", default_handler=str))
    return {"rows": int(len(df)), "columns": cols, "sample_rows": sample,
            "suggested_pitch_type": suggest_pitch_type(df)}
