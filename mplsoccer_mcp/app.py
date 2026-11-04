"""HTTP application: MCP Streamable HTTP endpoint + upload page + chart downloads + optional API key."""

from __future__ import annotations

import hmac
import html
import json
import os

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from . import data as D
from .server import get_chart, mcp

API_KEY = os.getenv("MCP_API_KEY", "").strip()
PUBLIC_PATHS = {"/health"}

UPLOAD_PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>mplsoccer MCP · Subir datos</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;color:#1f2a2e}
h1{font-size:1.4rem}form{display:grid;gap:12px;padding:20px;border:1px solid #ccd;border-radius:10px}
button{padding:10px;border:0;border-radius:8px;background:#1b998b;color:#fff;font-size:1rem;cursor:pointer}
pre{background:#f4f6f5;padding:12px;border-radius:8px;white-space:pre-wrap}</style></head><body>
<h1>Subir un dataset / Upload a dataset</h1>
<p>CSV, Excel (.xlsx), JSON (incl. StatsBomb events). Después dile al asistente el <b>dataset_id</b>.</p>
<form method="post" enctype="multipart/form-data" action="upload%(key)s">
<label>Nombre / name <input name="name" required placeholder="partido_fecha_5"></label>
<input type="file" name="file" required>
<button type="submit">Subir / Upload</button></form>%(result)s</body></html>"""


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return True
    supplied = request.headers.get("x-api-key") or request.query_params.get("key") or ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = supplied or auth[7:].strip()
    return bool(supplied) and hmac.compare_digest(supplied, API_KEY)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/charts/{chart_id}.png", methods=["GET"])
async def chart_png(request: Request) -> Response:
    png = await run_in_threadpool(get_chart, request.path_params["chart_id"])
    if png is None:
        return JSONResponse({"error": "chart not found or expired"}, status_code=404)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


@mcp.custom_route("/upload", methods=["GET", "POST"])
async def upload(request: Request) -> Response:
    keyq = f"?key={html.escape(request.query_params['key'])}" if request.query_params.get("key") else ""
    if request.method == "GET":
        return HTMLResponse(UPLOAD_PAGE % {"key": keyq, "result": ""})
    wants_html = "text/html" in request.headers.get("accept", "")
    try:
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form(max_part_size=D.MAX_DOWNLOAD_BYTES)
            up = form.get("file")
            if up is None or isinstance(up, str):
                raise D.ToolError("Missing 'file' field.")
            raw = await up.read()
            fname = up.filename or "upload"
            name = str(form.get("name") or fname.rsplit(".", 1)[0])
            sheet = form.get("sheet")
        else:
            raw = await request.body()
            fname = request.query_params.get("filename", "upload")
            name = request.query_params.get("name", fname.rsplit(".", 1)[0])
            sheet = request.query_params.get("sheet")
        if len(raw) > D.MAX_DOWNLOAD_BYTES:
            raise D.ToolError("File too large.")
        fmt = request.query_params.get("format") or D._guess_format(fname, ctype, raw)
        df = await run_in_threadpool(lambda: D.finalize_df(D.parse_bytes(raw, fmt, sheet=sheet)))
        ds = await run_in_threadpool(D.STORE.put, df, name, f"upload:{fname}")
        info = {"dataset_id": ds.id, "name": ds.name, "rows": len(df), "columns": list(df.columns)[:80],
                "suggested_pitch_type": D.suggest_pitch_type(df)}
    except D.ToolError as exc:
        if wants_html:
            return HTMLResponse(UPLOAD_PAGE % {"key": keyq, "result": f"<pre>Error: {html.escape(str(exc))}</pre>"},
                                status_code=400)
        return JSONResponse({"error": str(exc)}, status_code=400)
    if wants_html:
        msg = (f"<p>✅ Listo. Dile al asistente: <b>usa el dataset {html.escape(ds.id)}</b></p>"
               f"<pre>{html.escape(json.dumps(info, ensure_ascii=False, indent=2))}</pre>")
        return HTMLResponse(UPLOAD_PAGE % {"key": keyq, "result": msg})
    return JSONResponse(info)


class AuthMiddleware:
    """Pure ASGI middleware (keeps SSE streaming intact) enforcing MCP_API_KEY when set."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and API_KEY and scope.get("method") != "OPTIONS" \
                and scope.get("path") not in PUBLIC_PATHS and not scope.get("path", "").startswith("/charts/"):
            request = Request(scope)
            if not _authorized(request):
                resp = JSONResponse({"error": "unauthorized: send Authorization: Bearer <key>, X-API-Key, "
                                              "or ?key=<key>"}, status_code=401)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app():
    from starlette.middleware.cors import CORSMiddleware

    max_body = int(os.getenv("MPLSOCCER_MCP_MAX_BODY_MB", "25")) * 1024 * 1024
    inner = mcp.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, json_response=False,
                                    max_request_body_size=max_body, host="0.0.0.0")
    app = AuthMiddleware(inner)
    return CORSMiddleware(app, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                          allow_headers=["*"], expose_headers=["Mcp-Session-Id"])


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
