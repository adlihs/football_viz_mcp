"""Composable pitch charts: a figure is a pitch plus an ordered list of layers.

Every mplsoccer pitch plotting method is exposed as a layer type, so the LLM can
combine them freely (e.g. a KDE heatmap + arrows + labelled scatter), facet the
figure into small multiples, and style everything with JSON-only options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mplsoccer import (arrowhead_marker, football_left_boot_marker, football_right_boot_marker,
                       football_shirt_marker)
from pydantic import BaseModel, Field

from .data import Filter, ToolError, apply_filters, auto_column, finalize_df, get_df, numeric, resolve_column
from .render import (RENDER_LOCK, PitchConfig, category_colors, check_color, check_colormap, fig_to_png,
                     is_continuous, make_pitch, ordered_categories, scale_sizes, theme_of, add_texts)

LayerType = Literal[
    "scatter", "arrows", "lines", "comet", "plot", "kde", "hexbin", "heatmap", "heatmap_positional",
    "heatmap_zones", "zones", "flow", "sonar", "pass_network", "voronoi", "delaunay", "convexhull",
    "goal_angle", "text", "annotate", "polygon", "formation",
]

MARKERS = {
    "football": "football",
    "boot_left": football_left_boot_marker,
    "boot_right": football_right_boot_marker,
    "shirt": football_shirt_marker,
    "arrowhead": arrowhead_marker,
}


class Layer(BaseModel):
    """One drawing layer on the pitch. Layers are drawn in order (first = bottom)."""

    type: LayerType = Field(description="What to draw. See get_reference(topic='layers') for every option.")
    dataset_id: str | None = Field(default=None, description="Dataset for this layer (defaults to the chart's).")
    data: list[dict[str, Any]] | None = Field(default=None, description="Inline records for this layer only.")
    filters: list[Filter] | None = Field(default=None, description="Row filters for this layer (AND-combined).")
    x: str | None = Field(default=None, description="Column with x / start x (auto-detected if omitted).")
    y: str | None = Field(default=None, description="Column with y / start y (auto-detected if omitted).")
    end_x: str | None = Field(default=None, description="Column with end x (arrows, lines, flow, sonar, network).")
    end_y: str | None = Field(default=None, description="Column with end y.")
    color_by: str | None = Field(default=None, description="Column for colour: categories get a legend, numbers "
                                                           "a colormap + colorbar.")
    size_by: str | None = Field(default=None, description="Numeric column scaling marker size (e.g. xG).")
    label_by: str | None = Field(default=None, description="Column with text to write next to each point.")
    group_by: str | None = Field(default=None, description="Column defining groups (teams for voronoi, players for "
                                                           "convexhull/plot paths, passer for pass_network).")
    values: str | None = Field(default=None, description="Numeric column aggregated by heatmaps "
                                                         "(with statistic mean/sum/...).")
    color: str | None = Field(default=None, description="Single colour (name or hex).")
    color_map: dict[str, str] | None = Field(default=None, description="Explicit colours per category of color_by, "
                                                                       "e.g. {'Goal':'#D7263D','Saved':'#888'}.")
    cmap: str | None = Field(default=None, description="Matplotlib colormap for continuous colours/heatmaps.")
    alpha: float | None = Field(default=None, ge=0, le=1)
    size: float | None = Field(default=None, description="Marker size / line width / font size depending on type.")
    marker: str | None = Field(default=None, description="Matplotlib marker ('o','s','^','*','h', 'X') or "
                                                         "'football', 'boot_left', 'boot_right', 'shirt', 'arrowhead'.")
    label: str | None = Field(default=None, description="Legend label when color_by is not used.")
    mirror: bool = Field(default=False, description="Flip coordinates so the team attacks the other way.")
    zorder: float | None = None
    options: dict[str, Any] = Field(default_factory=dict, description="Type-specific options (see get_reference).")
    style: dict[str, Any] = Field(default_factory=dict, description="Extra matplotlib keyword arguments passed "
                                                                   "straight to the underlying call.")


class FigureOptions(BaseModel):
    title: str | None = None
    subtitle: str | None = None
    endnote: str | None = Field(default=None, description="Small text bottom-right (source, author, @handle).")
    theme: Literal["light", "dark", "grass", "night"] = "light"
    background: str | None = Field(default=None, description="Override figure background colour.")
    figheight: float = Field(default=9, ge=3, le=24, description="Figure height in inches.")
    dpi: int = Field(default=150, ge=50, le=300)
    legend: bool = True
    legend_loc: str = Field(default="upper left", description="Matplotlib legend location, or 'below'.")
    colorbar: bool = True
    colorbar_label: str | None = None
    facet_by: str | None = Field(default=None, description="Column of the chart dataset: draws one pitch per value "
                                                           "(small multiples).")
    facet_order: list[Any] | None = None
    ncols: int | None = Field(default=None, ge=1, le=8)
    max_facets: int = Field(default=16, ge=1, le=36)
    font_scale: float = Field(default=1.0, ge=0.5, le=3)


# --------------------------------------------------------------------------- #
# context helpers
# --------------------------------------------------------------------------- #


@dataclass
class Ctx:
    pitch: Any
    theme: dict[str, Any]
    fs: float
    handles: dict[str, Any] = field(default_factory=dict)
    mappables: list[tuple[Any, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_handle(self, label: Any, handle: Any) -> None:
        key = str(label)
        if key and key not in self.handles:
            self.handles[key] = handle


def _extent(pitch) -> tuple[float, float, float, float]:
    d = pitch.dim
    return min(d.left, d.right), max(d.left, d.right), min(d.bottom, d.top), max(d.bottom, d.top)


def _xy(ctx: Ctx, df: pd.DataFrame, xc: str, yc: str, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    x, y = numeric(df, xc), numeric(df, yc)
    x0, x1, y0, y1 = _extent(ctx.pitch)
    if mirror:
        x, y = x0 + x1 - x, y0 + y1 - y
    finite = ~(np.isnan(x) | np.isnan(y))
    if finite.any():
        tol_x, tol_y = (x1 - x0) * 0.03, (y1 - y0) * 0.03
        out = ((x < x0 - tol_x) | (x > x1 + tol_x) | (y < y0 - tol_y) | (y > y1 + tol_y)) & finite
        if out.mean() > 0.05:
            ctx.warnings.append(
                f"{int(out.sum())} points of '{xc}/{yc}' fall outside the pitch ({x0:g}-{x1:g} x {y0:g}-{y1:g}); "
                "check pitch_type (use describe_dataset for a suggestion).")
    return x, y


def _to_ax(ctx: Ctx, x: float, y: float) -> tuple[float, float]:
    from mplsoccer import VerticalPitch
    return (y, x) if isinstance(ctx.pitch, VerticalPitch) else (x, y)


def _base_kw(L: Layer, **defaults) -> dict[str, Any]:
    kw = dict(defaults)
    if L.alpha is not None:
        kw["alpha"] = L.alpha
    if L.zorder is not None:
        kw["zorder"] = L.zorder
    kw.update(L.style or {})
    return kw


def _color_groups(ctx: Ctx, df: pd.DataFrame, L: Layer, default_color: str | None = None):
    """Yield (mask, colour, label) for categorical colour, or a continuous spec."""
    if L.color_by:
        col = resolve_column(df, L.color_by, "color_by")
        s = df[col]
        if is_continuous(s) and not L.color_map:
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            return {"continuous": True, "values": vals, "cmap": check_colormap(L.cmap, ctx.theme["cmap"]),
                    "label": col}
        cats = ordered_categories(s, L.options.get("order"))
        cmap = category_colors(cats, ctx.theme, L.color_map)
        groups = [((s == c).to_numpy(), cmap[c], str(c)) for c in cats]
        nan_mask = s.isna().to_numpy()
        if nan_mask.any():
            groups.append((nan_mask, ctx.theme["muted"], "(empty)"))
        return {"continuous": False, "groups": groups, "label": col}
    color = check_color(L.color) or default_color or ctx.theme["palette"][0]
    return {"continuous": False, "groups": [(np.ones(len(df), bool), color, L.label or "")], "label": None}


def _label_points(ctx: Ctx, ax, df: pd.DataFrame, x: np.ndarray, y: np.ndarray, L: Layer) -> None:
    if not L.label_by:
        return
    col = resolve_column(df, L.label_by, "label_by")
    fs = float(L.options.get("label_size", 9)) * ctx.fs
    color = L.options.get("label_color", ctx.theme["text"])
    offset = float(L.options.get("label_offset", 8))
    for xi, yi, t in zip(x, y, df[col].astype(str)):
        if np.isnan(xi) or np.isnan(yi) or t in ("nan", "None"):
            continue
        ax.annotate(t, xy=_to_ax(ctx, xi, yi), xytext=(0, offset), textcoords="offset points", ha="center",
                    va="bottom", fontsize=fs, color=color, zorder=10)


# --------------------------------------------------------------------------- #
# layer implementations
# --------------------------------------------------------------------------- #


def _scatter(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    marker = MARKERS.get(L.marker or "o", L.marker or "o")
    base_size = L.size or 150
    if L.size_by:
        sc = resolve_column(df, L.size_by, "size_by")
        rng = L.options.get("size_range", [40, 900])
        sizes = scale_sizes(numeric(df, sc), (float(rng[0]), float(rng[1])), L.options.get("size_max"))
    else:
        sizes = np.full(len(df), float(base_size))
    rot = None
    if L.options.get("rotation_by"):
        rot = numeric(df, resolve_column(df, L.options["rotation_by"], "rotation_by"))
    edge = L.options.get("edgecolor", ctx.theme["background"] if marker != "football" else None)
    kw = _base_kw(L, zorder=4, linewidths=L.options.get("linewidth", 1.0))
    if edge and marker != "football":
        kw.setdefault("edgecolors", edge)
    spec = _color_groups(ctx, df, L)
    n = 0
    if spec["continuous"]:
        sc_kw = dict(kw, c=spec["values"], cmap=spec["cmap"])
        for k in ("vmin", "vmax"):
            if k in L.options:
                sc_kw[k] = L.options[k]
        art = ctx.pitch.scatter(x, y, s=sizes, marker=marker, ax=ax,
                                rotation_degrees=rot, **sc_kw)
        ctx.mappables.append((art, spec["label"]))
        n = int(np.isfinite(x).sum())
    else:
        for mask, color, label in spec["groups"]:
            m = mask & np.isfinite(x) & np.isfinite(y)
            if not m.any():
                continue
            gkw = dict(kw)
            if marker == "football":
                gkw.pop("edgecolors", None)
            else:
                gkw.setdefault("color", color)
            ctx.pitch.scatter(x[m], y[m], s=sizes[m], marker=marker, ax=ax,
                              rotation_degrees=None if rot is None else rot[m], **gkw)
            n += int(m.sum())
            if label:
                ctx.add_handle(label, Line2D([], [], marker="o", linestyle="", markersize=9, color=color,
                                             markeredgecolor=edge or color))
    _label_points(ctx, ax, df, x, y, L)
    if L.size_by and L.options.get("size_legend", True):
        ctx.notes.append(f"marker size ∝ {L.size_by}")
    return f"scatter: {n} points"


def _segments(ctx: Ctx, df: pd.DataFrame, L: Layer):
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    exc, eyc = auto_column(df, L.end_x, "end_x"), auto_column(df, L.end_y, "end_y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ex, ey = _xy(ctx, df, exc, eyc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ex) & np.isfinite(ey)
    return x, y, ex, ey, ok


def _arrows(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    x, y, ex, ey, ok = _segments(ctx, df, L)
    w = L.size or 2
    kw = _base_kw(L, width=w, headwidth=L.options.get("headwidth", 5), headlength=L.options.get("headlength", 5),
                  headaxislength=L.options.get("headaxislength", 4.5), zorder=3)
    spec = _color_groups(ctx, df, L)
    n = 0
    if spec["continuous"]:
        vals = spec["values"]
        art = ctx.pitch.arrows(x[ok], y[ok], ex[ok], ey[ok], vals[ok], cmap=spec["cmap"], ax=ax, **kw)
        ctx.mappables.append((art, spec["label"]))
        n = int(ok.sum())
    else:
        for mask, color, label in spec["groups"]:
            m = mask & ok
            if not m.any():
                continue
            ctx.pitch.arrows(x[m], y[m], ex[m], ey[m], color=color, ax=ax, **kw)
            n += int(m.sum())
            if label:
                ctx.add_handle(label, Line2D([], [], color=color, lw=2.5, marker=">", markersize=7))
    if L.options.get("show_start", False):
        ctx.pitch.scatter(x[ok], y[ok], s=L.options.get("start_size", 25), color=L.options.get("start_color",
                          ctx.theme["text"]), ax=ax, zorder=kw.get("zorder", 3) + 0.1)
    _label_points(ctx, ax, df[ok], x[ok], y[ok], L)
    return f"arrows: {n}"


def _lines(ctx: Ctx, ax, df: pd.DataFrame, L: Layer, comet: bool = False) -> str:
    x, y, ex, ey, ok = _segments(ctx, df, L)
    comet = comet or bool(L.options.get("comet", False))
    kw = _base_kw(L, lw=L.size or (3 if comet else 2), zorder=3)
    kw.setdefault("comet", comet)
    kw.setdefault("transparent", bool(L.options.get("transparent", comet)))
    if "alpha_start" in L.options:
        kw["alpha_start"] = L.options["alpha_start"]
    if "alpha_end" in L.options:
        kw["alpha_end"] = L.options["alpha_end"]
    if kw.get("transparent"):
        kw.pop("alpha", None)
    spec = _color_groups(ctx, df, L)
    n = 0
    if spec["continuous"]:
        from matplotlib import cm as mcm
        from matplotlib import colors as mcolors
        vals = spec["values"][ok]
        norm = mcolors.Normalize(vmin=L.options.get("vmin", np.nanmin(vals)), vmax=L.options.get("vmax", np.nanmax(vals)))
        sm = mcm.ScalarMappable(norm=norm, cmap=spec["cmap"])
        ctx.pitch.lines(x[ok], y[ok], ex[ok], ey[ok], color=sm.to_rgba(np.nan_to_num(vals)), ax=ax, **kw)
        ctx.mappables.append((sm, spec["label"]))
        n = int(ok.sum())
    else:
        for mask, color, label in spec["groups"]:
            m = mask & ok
            if not m.any():
                continue
            ctx.pitch.lines(x[m], y[m], ex[m], ey[m], color=color, ax=ax, **kw)
            n += int(m.sum())
            if label:
                ctx.add_handle(label, Line2D([], [], color=color, lw=3))
    if L.options.get("show_end", comet):
        ctx.pitch.scatter(ex[ok], ey[ok], s=L.options.get("end_size", 40), color=L.options.get(
            "end_color", ctx.theme["text"]), ax=ax, zorder=kw.get("zorder", 3) + 0.1,
            edgecolors=ctx.theme["background"], linewidths=0.8)
    return f"{'comet ' if comet else ''}lines: {n}"


def _plot(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    if L.options.get("sort_by"):
        df = df.sort_values(resolve_column(df, L.options["sort_by"], "sort_by"))
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    kw = _base_kw(L, lw=L.size or 2, zorder=3)
    if L.marker:
        kw.setdefault("marker", L.marker)
    by = L.group_by or L.color_by
    if by:
        col = resolve_column(df, by, "group_by")
        cats = ordered_categories(df[col], L.options.get("order"))
        cmap = category_colors(cats, ctx.theme, L.color_map)
        for c in cats:
            m = (df[col] == c).to_numpy()
            ctx.pitch.plot(x[m], y[m], color=cmap[c], ax=ax, **kw)
            ctx.add_handle(c, Line2D([], [], color=cmap[c], lw=3))
        return f"plot: {len(cats)} paths"
    color = check_color(L.color) or ctx.theme["palette"][0]
    ctx.pitch.plot(x, y, color=color, ax=ax, **kw)
    if L.label:
        ctx.add_handle(L.label, Line2D([], [], color=color, lw=3))
    _label_points(ctx, ax, df, x, y, L)
    return f"plot: {len(df)} points"


def _kde(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise ToolError("kde needs at least 3 points.")
    o = L.options
    kw = _base_kw(L, fill=o.get("fill", True), levels=o.get("levels", 100), thresh=o.get("thresh", 0.05),
                  cut=o.get("cut", 4), bw_adjust=o.get("bw_adjust", 1), zorder=1)
    if L.color and not L.cmap:
        kw["color"] = check_color(L.color)
    else:
        kw["cmap"] = check_colormap(L.cmap, ctx.theme["cmap"])
    ctx.pitch.kdeplot(x[ok], y[ok], ax=ax, **kw)
    return f"kde: {int(ok.sum())} points"


def _hexbin(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    o = L.options
    gs = o.get("gridsize", [10, 8])
    kw = _base_kw(L, gridsize=tuple(gs) if isinstance(gs, list) else gs, cmap=check_colormap(L.cmap, ctx.theme["cmap"]),
                  edgecolors=o.get("edgecolor", ctx.theme["background"]), mincnt=o.get("mincnt", 1), zorder=1)
    if L.values:
        kw["C"] = numeric(df, resolve_column(df, L.values, "values"))[ok]
        kw["reduce_C_function"] = {"mean": np.mean, "sum": np.sum, "median": np.median,
                                   "max": np.max, "min": np.min}[o.get("statistic", "mean")]
    art = ctx.pitch.hexbin(x[ok], y[ok], ax=ax, **kw)
    ctx.mappables.append((art, L.values or "count"))
    return f"hexbin: {int(ok.sum())} points"


def _stat_values(df: pd.DataFrame, L: Layer, ok: np.ndarray):
    stat = L.options.get("statistic", "count" if not L.values else "mean")
    vals = numeric(df, resolve_column(df, L.values, "values"))[ok] if L.values else None
    if stat != "count" and vals is None:
        raise ToolError(f"statistic='{stat}' needs a 'values' column.")
    return stat, vals


def _label_stats(ctx: Ctx, ax, stats, L: Layer, normalize: bool) -> None:
    o = L.options
    if not o.get("show_labels", False):
        return
    fmt = o.get("label_format") or ("{:.0%}" if normalize else ("{:.0f}" if o.get("statistic", "count") == "count"
                                                                 else "{:.2f}"))
    ctx.pitch.label_heatmap(stats, color=o.get("label_color", ctx.theme["text"]),
                            fontsize=float(o.get("label_size", 14)) * ctx.fs, ax=ax, ha="center", va="center",
                            str_format=fmt, exclude_zeros=o.get("exclude_zeros", True),
                            exclude_nan=True, zorder=6, path_effects=_stroke(ctx))


def _stroke(ctx: Ctx):
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=2.5, foreground=ctx.theme["background"]
                                   if ctx.theme["pitch_color"] != "grass" else "#000000")]


def _heatmap(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    o = L.options
    stat, vals = _stat_values(df, L, ok)
    bins = o.get("bins", [6, 5])
    bins = tuple(bins) if isinstance(bins, list) else bins
    normalize = bool(o.get("normalize", False))
    stats = ctx.pitch.bin_statistic(x[ok], y[ok], values=vals, statistic=stat, bins=bins, normalize=normalize)
    if o.get("smooth"):
        from scipy.ndimage import gaussian_filter
        stats["statistic"] = gaussian_filter(np.nan_to_num(stats["statistic"]), float(o["smooth"]))
    kw = _base_kw(L, cmap=check_colormap(L.cmap, ctx.theme["cmap"]),
                  edgecolors=o.get("edgecolor", ctx.theme["background"]), zorder=1)
    for k in ("vmin", "vmax"):
        if k in o:
            kw[k] = o[k]
    art = ctx.pitch.heatmap(stats, ax=ax, **kw)
    ctx.mappables.append((art, ("% of total" if normalize else (L.values or "count"))))
    _label_stats(ctx, ax, stats, L, normalize)
    return f"heatmap: {int(ok.sum())} points in {bins} bins ({stat})"


def _heatmap_positional(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    o = L.options
    stat, vals = _stat_values(df, L, ok)
    normalize = bool(o.get("normalize", False))
    stats = ctx.pitch.bin_statistic_positional(x[ok], y[ok], values=vals, statistic=stat,
                                               positional=o.get("positional", "full"), normalize=normalize)
    kw = _base_kw(L, cmap=check_colormap(L.cmap, ctx.theme["cmap"]),
                  edgecolors=o.get("edgecolor", ctx.theme["background"]), zorder=1)
    for k in ("vmin", "vmax"):
        if k in o:
            kw[k] = o[k]
    arts = ctx.pitch.heatmap_positional(stats, ax=ax, **kw)
    arts = arts if isinstance(arts, (list, tuple)) else [arts]
    if arts:
        ctx.mappables.append((arts[0], ("% of total" if normalize else (L.values or "count"))))
    _label_stats(ctx, ax, stats, L, normalize)
    return f"positional heatmap ({o.get('positional', 'full')}): {int(ok.sum())} points"


def _zones(ctx: Ctx, L: Layer):
    z = L.options.get("zones", "positional_full")
    if isinstance(z, str):
        kind = z.replace("positional_", "") or "full"
        if kind not in ("full", "horizontal", "vertical"):
            raise ToolError("options.zones must be 'positional_full', 'positional_horizontal', "
                            "'positional_vertical' or a list of [x0, x1, y0, y1] rectangles.")
        zones, names = ctx.pitch.positional_zones(kind)
        return list(zones), list(L.options.get("names") or names)
    zones = [tuple(map(float, r)) for r in z]
    names = L.options.get("names") or [f"Z{i + 1}" for i in range(len(zones))]
    return zones, list(names)


def _heatmap_zones(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    o = L.options
    stat, vals = _stat_values(df, L, ok)
    zones, names = _zones(ctx, L)
    normalize = bool(o.get("normalize", False))
    try:
        stats = ctx.pitch.bin_statistic_zones(x[ok], y[ok], zones, values=vals, statistic=stat,
                                              normalize=normalize, names=names)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Zones heatmap failed (zones must tile the whole pitch exactly): {exc}") from exc
    kw = _base_kw(L, cmap=check_colormap(L.cmap, ctx.theme["cmap"]),
                  edgecolors=o.get("edgecolor", ctx.theme["background"]), zorder=1)
    art = ctx.pitch.heatmap_zones(stats, ax=ax, **kw)
    if art is not None:
        ctx.mappables.append((art if not isinstance(art, list) else art[0],
                              ("% of total" if normalize else (L.values or "count"))))
    if o.get("show_labels", False):
        fmt = o.get("label_format") or ("{:.0%}" if normalize else "{:.0f}")
        stat_arr = np.asarray(stats["statistic"], dtype=float)
        for (x0, x1, y0, y1), v in zip(zones, stat_arr):
            if np.isnan(v) or (v == 0 and o.get("exclude_zeros", True)):
                continue
            ctx.pitch.text((x0 + x1) / 2, (y0 + y1) / 2, fmt.format(v), ax=ax, ha="center", va="center",
                           color=o.get("label_color", ctx.theme["text"]),
                           fontsize=float(o.get("label_size", 13)) * ctx.fs, zorder=6,
                           path_effects=_stroke(ctx))
    return f"zone heatmap: {int(ok.sum())} points in {len(zones)} zones ({stat})"


def _draw_zones(ctx: Ctx, ax, df: pd.DataFrame | None, L: Layer) -> str:
    zones, names = _zones(ctx, L)
    o = L.options
    ctx.pitch.draw_zones(zones, names=names, facecolor=o.get("facecolor", "none"),
                         edgecolor=check_color(L.color) or ctx.theme["muted"], alpha=L.alpha if L.alpha is not None else 0.8,
                         label=o.get("show_names", True), ax=ax, zorder=L.zorder or 2, **(L.style or {}))
    return f"zones: {len(zones)}"


def _flow(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    x, y, ex, ey, ok = _segments(ctx, df, L)
    o = L.options
    bins = o.get("bins", [6, 4])
    kw = _base_kw(L, bins=tuple(bins) if isinstance(bins, list) else bins,
                  arrow_type=o.get("arrow_type", "scale"), arrow_length=o.get("arrow_length", 15),
                  zorder=3)
    if L.cmap:
        kw["cmap"] = check_colormap(L.cmap, "Blues")
    else:
        kw["color"] = check_color(L.color) or ctx.theme["text"]
    if "headwidth" in o:
        kw["headwidth"] = o["headwidth"]
    ctx.pitch.flow(x[ok], y[ok], ex[ok], ey[ok], ax=ax, **kw)
    return f"flow: {int(ok.sum())} movements"


def _sonar(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    o = L.options
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    if o.get("angle_column"):
        angle = numeric(df, resolve_column(df, o["angle_column"], "angle_column"))
        if o.get("angle_degrees", False):
            angle = np.deg2rad(angle)
        distance = None
    else:
        exc, eyc = auto_column(df, L.end_x, "end_x"), auto_column(df, L.end_y, "end_y")
        ex, ey = _xy(ctx, df, exc, eyc, L.mirror)
        angle, distance = ctx.pitch.calculate_angle_and_distance(x, y, ex, ey)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(angle)
    n_angles = int(o.get("angle_bins", 10))
    color_stat = o.get("color_by_stat", "distance")  # distance | values | none
    cvals = None
    if color_stat == "values" and L.values:
        cvals = numeric(df, resolve_column(df, L.values, "values"))
    elif color_stat == "distance" and distance is not None:
        cvals = distance
    cmap = check_colormap(L.cmap, "viridis")
    common = dict(cmap=cmap, ec=o.get("edgecolor", ctx.theme["background"]), lw=o.get("linewidth", 0.5),
                  sonar_alpha=L.alpha if L.alpha is not None else 1, rmin=o.get("rmin", 0))
    x0, x1, y0, y1 = _extent(ctx.pitch)
    scale = float(o.get("sonar_scale", 0.95))
    if o.get("zones"):
        zones, names = _zones(ctx, L)
        sl = ctx.pitch.bin_statistic_sonar_zones(x[ok], y[ok], angle[ok], zones, angle_bins=n_angles,
                                                 center=o.get("center", True))
        sc = None
        if cvals is not None:
            sc = ctx.pitch.bin_statistic_sonar_zones(x[ok], y[ok], angle[ok], zones, values=cvals[ok],
                                                     statistic="mean", angle_bins=n_angles,
                                                     center=o.get("center", True))
        size = min(min(z[1] - z[0], z[3] - z[2]) for z in zones) * scale
        if sc is None:
            common["cmap"] = None
            common["fc"] = check_color(L.color) or ctx.theme["palette"][0]
        ctx.pitch.sonar_zones(sl, stats_color=sc, ax=ax, width=size,
                              **{k: v for k, v in common.items() if v is not None}, **(L.style or {}))
        return f"sonar (zones): {int(ok.sum())} actions"
    bins = o.get("bins", [6, 4])
    bins = tuple(bins) + (n_angles,) if isinstance(bins, list) and len(bins) == 2 else tuple(bins)
    sl = ctx.pitch.bin_statistic_sonar(x[ok], y[ok], angle[ok], bins=bins, center=o.get("center", True))
    sc = None
    if cvals is not None:
        sc = ctx.pitch.bin_statistic_sonar(x[ok], y[ok], angle[ok], values=cvals[ok], statistic="mean",
                                           bins=bins, center=o.get("center", True))
    if sc is None:
        common["cmap"] = None
        common["fc"] = check_color(L.color) or ctx.theme["palette"][0]
    size = min((x1 - x0) / bins[0], (y1 - y0) / bins[1]) * scale
    ctx.pitch.sonar_grid(sl, stats_color=sc, ax=ax, width=size,
                         **{k: v for k, v in common.items() if v is not None},
                         **(L.style or {}))
    if sc is not None:
        ctx.notes.append(f"sonar colour = mean {'distance' if color_stat == 'distance' else L.values}")
    return f"sonar grid {bins}: {int(ok.sum())} actions"


def _pass_network(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    o = L.options
    passer = resolve_column(df, L.group_by or o.get("passer"), "passer (group_by)")
    recip = resolve_column(df, o.get("recipient"), "options.recipient")
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    d = pd.DataFrame({"passer": df[passer].to_numpy(), "recipient": df[recip].to_numpy(), "x": x, "y": y})
    exc = auto_column(df, L.end_x, "end_x", required=False)
    eyc = auto_column(df, L.end_y, "end_y", required=False)
    if exc and eyc:
        ex, ey = _xy(ctx, df, exc, eyc, L.mirror)
        d["ex"], d["ey"] = ex, ey
    d = d.dropna(subset=["passer", "x", "y"])
    # average positions: where each player passed from (+ where they received, if end coords exist)
    locs = [d[["passer", "x", "y"]].rename(columns={"passer": "player"})]
    if "ex" in d:
        locs.append(d.dropna(subset=["recipient"])[["recipient", "ex", "ey"]].rename(
            columns={"recipient": "player", "ex": "x", "ey": "y"}))
    allpos = pd.concat(locs).dropna()
    avg = allpos.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), touches=("x", "size"))
    passes_made = d.groupby("passer").size()
    avg["passes"] = passes_made.reindex(avg.index).fillna(0)
    players = o.get("players")
    if players:
        keep = {str(p).lower() for p in players}
        avg = avg[[str(i).lower() in keep for i in avg.index]]
    between = d.dropna(subset=["recipient"]).copy()
    if o.get("undirected", True):
        between["a"] = between[["passer", "recipient"]].astype(str).min(axis=1)
        between["b"] = between[["passer", "recipient"]].astype(str).max(axis=1)
    else:
        between["a"], between["b"] = between["passer"].astype(str), between["recipient"].astype(str)
    edges = between.groupby(["a", "b"]).size().reset_index(name="count")
    min_passes = int(o.get("min_passes", 3))
    edges = edges[(edges["count"] >= min_passes) & (edges["a"] != edges["b"])]
    idx = {str(k): k for k in avg.index}
    edges = edges[edges["a"].isin(idx) & edges["b"].isin(idx)]
    line_color = check_color(o.get("edge_color")) or check_color(L.color) or ctx.theme["text"]
    node_color = check_color(o.get("node_color")) or ctx.theme["palette"][0]
    if not edges.empty:
        maxc = edges["count"].max()
        widths = (edges["count"] / maxc * float(o.get("max_edge_width", 14))).clip(lower=1).to_numpy()
        a = avg.loc[[idx[v] for v in edges["a"]]]
        b = avg.loc[[idx[v] for v in edges["b"]]]
        from matplotlib import colors as mcolors
        rgba = np.array(mcolors.to_rgba(line_color))
        alphas = (0.25 + 0.75 * edges["count"].to_numpy() / maxc) if o.get("edge_alpha_by_count", True) else \
            np.full(len(edges), 0.7)
        cols = np.tile(rgba, (len(edges), 1))
        cols[:, 3] = alphas
        ctx.pitch.lines(a["x"].to_numpy(), a["y"].to_numpy(), b["x"].to_numpy(), b["y"].to_numpy(),
                        lw=widths, color=cols, zorder=2, ax=ax)
    rng = o.get("node_size_range", [300, 1800])
    sizes = scale_sizes(avg["passes"].to_numpy(dtype=float), (float(rng[0]), float(rng[1])))
    ctx.pitch.scatter(avg["x"], avg["y"], s=sizes, color=node_color, edgecolors=o.get("node_edge_color",
                      ctx.theme["text"]), linewidth=1.5, alpha=1, zorder=3, ax=ax)
    labels = o.get("labels", {})  # mapping player -> short label
    lab_kind = o.get("label", "name")  # name | last_name | initials | none
    for player, row in avg.iterrows():
        txt = str(labels.get(str(player), player))
        if str(player) not in labels:
            if lab_kind == "last_name":
                txt = txt.split()[-1]
            elif lab_kind == "initials":
                txt = "".join(p[0] for p in txt.split() if p)
            elif lab_kind == "none":
                continue
        inside = lab_kind in ("initials",) or o.get("label_inside", False)
        if inside:
            ctx.pitch.annotate(txt, xy=(row.x, row.y), ax=ax, ha="center", va="center",
                               color=o.get("label_color", ctx.theme["background"]),
                               fontsize=float(o.get("label_size", 10)) * ctx.fs, fontweight="bold", zorder=4)
        else:
            ax.annotate(txt, xy=_to_ax(ctx, row.x, row.y), xytext=(0, -(math.sqrt(sizes.max()) / 2 + 9)),
                        textcoords="offset points", ha="center", va="center",
                        color=o.get("label_color", ctx.theme["text"]),
                        fontsize=float(o.get("label_size", 10)) * ctx.fs, zorder=4)
    top = edges.sort_values("count", ascending=False).head(5)
    combos = ", ".join(f"{r.a}–{r.b} ({r['count']})" for _, r in top.iterrows())
    ctx.notes.append(f"pass network: node size = passes made; edges shown with ≥{min_passes} passes")
    return f"pass network: {len(avg)} players, {len(edges)} connections; top: {combos or 'none'}"


def _team_mask(ctx: Ctx, df: pd.DataFrame, L: Layer):
    col = resolve_column(df, L.group_by or L.color_by or L.options.get("team"), "team (group_by)")
    cats = ordered_categories(df[col], L.options.get("order"))
    team_a = L.options.get("team_a")
    if team_a is None:
        team_a = cats[0]
    mask = (df[col].astype(str).str.lower() == str(team_a).lower()).to_numpy()
    other = [c for c in cats if str(c).lower() != str(team_a).lower()]
    colors = category_colors([team_a] + other, ctx.theme, L.color_map)
    return col, mask, team_a, (other[0] if other else "other"), colors


def _voronoi(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    df = df[ok]
    x, y = x[ok], y[ok]
    col, mask, a, b, colors = _team_mask(ctx, df, L)
    t1, t2 = ctx.pitch.voronoi(x, y, mask)
    alpha = L.alpha if L.alpha is not None else 0.3
    ec = L.options.get("edgecolor", ctx.theme["background"])
    ctx.pitch.polygon(t1, ax=ax, fc=colors[a], ec=ec, lw=1, alpha=alpha, zorder=1)
    ctx.pitch.polygon(t2, ax=ax, fc=colors.get(b, ctx.theme["palette"][1]), ec=ec, lw=1, alpha=alpha, zorder=1)
    ctx.add_handle(a, Patch(color=colors[a]))
    ctx.add_handle(b, Patch(color=colors.get(b, ctx.theme["palette"][1])))
    if L.options.get("show_players", True):
        for m, team in ((mask, a), (~mask, b)):
            ctx.pitch.scatter(x[m], y[m], s=L.size or 160, color=colors.get(team, ctx.theme["palette"][1]),
                              edgecolors=ctx.theme["text"], linewidth=1, zorder=4, ax=ax)
    _label_points(ctx, ax, df, x, y, L)
    return f"voronoi: {int(mask.sum())} vs {int((~mask).sum())} players"


def _delaunay(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    groups = [(np.ones(len(df), bool), check_color(L.color) or ctx.theme["palette"][0], L.label or "")]
    if L.group_by:
        col = resolve_column(df, L.group_by, "group_by")
        cats = ordered_categories(df[col])
        cm = category_colors(cats, ctx.theme, L.color_map)
        groups = [((df[col] == c).to_numpy(), cm[c], str(c)) for c in cats]
    n = 0
    for m, color, label in groups:
        m = m & ok
        if m.sum() < 3:
            continue
        ctx.pitch.triplot(x[m], y[m], ax=ax, color=color, lw=L.size or 1.5, alpha=L.alpha or 0.8, zorder=2,
                          **(L.style or {}))
        n += 1
        if label:
            ctx.add_handle(label, Line2D([], [], color=color, lw=2))
    return f"delaunay triangulation: {n} group(s)"


def _convexhull(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    groups = [(np.ones(len(df), bool), check_color(L.color) or ctx.theme["palette"][0], L.label or "")]
    if L.group_by:
        col = resolve_column(df, L.group_by, "group_by")
        cats = ordered_categories(df[col], L.options.get("order"))
        cm = category_colors(cats, ctx.theme, L.color_map)
        groups = [((df[col] == c).to_numpy(), cm[c], str(c)) for c in cats]
    n = 0
    for m, color, label in groups:
        m = m & ok
        if m.sum() < 3:
            continue
        xs, ys = x[m], y[m]
        trim = L.options.get("trim_outliers")
        if trim:  # drop points far from the centroid (keep given share)
            cx, cy = np.median(xs), np.median(ys)
            dist = np.hypot(xs - cx, ys - cy)
            keep = dist <= np.quantile(dist, float(trim))
            xs, ys = xs[keep], ys[keep]
        hull = ctx.pitch.convexhull(xs, ys)
        ctx.pitch.polygon(hull, ax=ax, fc=color, ec=color, lw=2, alpha=L.alpha if L.alpha is not None else 0.25,
                          zorder=2)
        if L.options.get("show_points", True):
            ctx.pitch.scatter(xs, ys, s=L.size or 40, color=color, zorder=3, ax=ax, edgecolors=ctx.theme["background"])
        n += 1
        if label:
            ctx.add_handle(label, Patch(color=color, alpha=0.6))
    return f"convex hull: {n} group(s)"


def _goal_angle(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    ok = np.isfinite(x) & np.isfinite(y)
    n = 0
    for xi, yi in list(zip(x[ok], y[ok]))[: int(L.options.get("max_shots", 30))]:
        ctx.pitch.goal_angle(xi, yi, ax=ax, goal=L.options.get("goal", "right"),
                             color=check_color(L.color) or ctx.theme["palette"][0],
                             alpha=L.alpha if L.alpha is not None else 0.3, zorder=L.zorder or 1.5)
        n += 1
    return f"goal angle: {n} shot(s)"


def _text(ctx: Ctx, ax, df: pd.DataFrame, L: Layer) -> str:
    xc, yc = auto_column(df, L.x, "x"), auto_column(df, L.y, "y")
    x, y = _xy(ctx, df, xc, yc, L.mirror)
    col = resolve_column(df, L.label_by or L.options.get("text_column"), "label_by")
    kw = _base_kw(L, ha="center", va="center", fontsize=(L.size or 11) * ctx.fs,
                  color=check_color(L.color) or ctx.theme["text"], zorder=6)
    n = 0
    for xi, yi, t in zip(x, y, df[col].astype(str)):
        if np.isfinite(xi) and np.isfinite(yi):
            ctx.pitch.text(xi, yi, t, ax=ax, **kw)
            n += 1
    return f"text: {n}"


def _annotate(ctx: Ctx, ax, df, L: Layer) -> str:
    items = L.options.get("annotations") or []
    if not items:
        raise ToolError("annotate needs options.annotations = [{text, x, y, text_x?, text_y?, arrow?}].")
    for it in items:
        xy = (float(it["x"]), float(it["y"]))
        kw = dict(ha="center", va="center", fontsize=float(it.get("size", L.size or 12)) * ctx.fs,
                  color=check_color(it.get("color") or L.color) or ctx.theme["text"], zorder=7)
        if it.get("bold"):
            kw["fontweight"] = "bold"
        if it.get("box", False):
            kw["bbox"] = dict(boxstyle="round,pad=0.3", fc=ctx.theme["background"], ec=kw["color"], alpha=0.85)
        if "text_x" in it and "text_y" in it:
            xyt = (float(it["text_x"]), float(it["text_y"]))
            if it.get("arrow", True):
                kw["arrowprops"] = dict(arrowstyle="->", color=kw["color"], lw=1.5,
                                        connectionstyle=f"arc3,rad={float(it.get('curve', 0.2))}")
            ctx.pitch.annotate(str(it["text"]), xy=xy, xytext=xyt, ax=ax, **kw)
        else:
            ctx.pitch.annotate(str(it["text"]), xy=xy, ax=ax, **kw)
    return f"annotations: {len(items)}"


def _polygon(ctx: Ctx, ax, df, L: Layer) -> str:
    polys = L.options.get("polygons") or ([L.options["vertices"]] if L.options.get("vertices") else None)
    if not polys:
        raise ToolError("polygon needs options.polygons = [[[x, y], [x, y], ...], ...] or options.vertices.")
    verts = [np.asarray(p, dtype=float) for p in polys]
    color = check_color(L.color) or ctx.theme["palette"][0]
    ctx.pitch.polygon(verts, ax=ax, fc=L.options.get("facecolor", color), ec=L.options.get("edgecolor", color),
                      alpha=L.alpha if L.alpha is not None else 0.3, lw=L.size or 2, zorder=L.zorder or 2)
    if L.label:
        ctx.add_handle(L.label, Patch(color=color, alpha=0.5))
    return f"polygons: {len(verts)}"


def _formation(ctx: Ctx, ax, df, L: Layer) -> str:
    o = L.options
    form = str(o.get("formation", "433")).replace("-", "")
    if form not in ctx.pitch.formations:
        raise ToolError(f"Unknown formation '{form}'. Valid: {ctx.pitch.formations}")
    positions = ctx.pitch.get_formation(form)
    flip, half = bool(o.get("flip", False)), bool(o.get("half", False))
    suffix = ("_half" if half else "") + ("_flip" if flip else "")
    xs = np.array([getattr(p, "x" + suffix) for p in positions])
    ys = np.array([getattr(p, "y" + suffix) for p in positions])
    names = [p.name for p in positions]
    players = o.get("players")
    if isinstance(players, dict):
        low = {str(k).upper(): v for k, v in players.items()}
        labels = [str(low.get(n, n)) for n in names]
    elif isinstance(players, list):
        labels = [str(players[i]) if i < len(players) else names[i] for i in range(len(names))]
    else:
        labels = names
    color = check_color(L.color) or ctx.theme["palette"][0]
    ctx.pitch.scatter(xs, ys, s=L.size or 900, color=color, edgecolors=o.get("edgecolor", ctx.theme["text"]),
                      linewidth=1.5, zorder=4, ax=ax, marker=MARKERS.get(L.marker or "o", L.marker or "o"))
    numbers = o.get("numbers")
    if numbers:
        nums = [str(numbers.get(n, "")) if isinstance(numbers, dict) else str(numbers[i]) for i, n in enumerate(names)]
        for xi, yi, t in zip(xs, ys, nums):
            ctx.pitch.text(xi, yi, t, ax=ax, ha="center", va="center", color=o.get("number_color", "white"),
                           fontsize=12 * ctx.fs, fontweight="bold", zorder=5)
    elif o.get("show_positions_inside", True) and players:
        for xi, yi, t in zip(xs, ys, names):
            ctx.pitch.text(xi, yi, t, ax=ax, ha="center", va="center", color=o.get("number_color", "white"),
                           fontsize=9 * ctx.fs, fontweight="bold", zorder=5)
    for xi, yi, t in zip(xs, ys, labels):
        if not players and not numbers:
            ctx.pitch.text(xi, yi, t, ax=ax, ha="center", va="center", color=o.get("number_color", "white"),
                           fontsize=10 * ctx.fs, fontweight="bold", zorder=5)
            continue
        ax.annotate(t, xy=_to_ax(ctx, xi, yi), xytext=(0, -(math.sqrt(L.size or 900) / 2 + 9)),
                    textcoords="offset points", ha="center", va="center", color=o.get("label_color", ctx.theme["text"]),
                    fontsize=float(o.get("label_size", 11)) * ctx.fs, zorder=5)
    if L.label:
        ctx.add_handle(L.label, Line2D([], [], marker="o", linestyle="", markersize=10, color=color))
    return f"formation {form}: {len(positions)} players"


IMPLS = {
    "scatter": _scatter, "arrows": _arrows, "lines": _lines,
    "comet": lambda c, a, d, L: _lines(c, a, d, L, comet=True), "plot": _plot, "kde": _kde, "hexbin": _hexbin,
    "heatmap": _heatmap, "heatmap_positional": _heatmap_positional, "heatmap_zones": _heatmap_zones,
    "zones": _draw_zones, "flow": _flow, "sonar": _sonar, "pass_network": _pass_network, "voronoi": _voronoi,
    "delaunay": _delaunay, "convexhull": _convexhull, "goal_angle": _goal_angle, "text": _text,
    "annotate": _annotate, "polygon": _polygon, "formation": _formation,
}
NO_DATA_LAYERS = {"annotate", "polygon", "formation", "zones"}


# --------------------------------------------------------------------------- #
# figure assembly
# --------------------------------------------------------------------------- #


def _layer_df(L: Layer, chart_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if L.data:
        df = finalize_df(pd.json_normalize(L.data, sep="."))
    elif L.dataset_id:
        df = get_df(L.dataset_id, None)
    else:
        df = chart_df
    if df is None:
        if L.type in NO_DATA_LAYERS:
            return None
        raise ToolError(f"Layer '{L.type}' needs data: set dataset_id/data on the chart or on the layer.")
    return apply_filters(df, L.filters)


def render_pitch_chart(*, pitch: PitchConfig | None, layers: list[Layer], dataset_id: str | None = None,
                       data: list[dict[str, Any]] | None = None, filters: list[Filter] | None = None,
                       figure: FigureOptions | None = None) -> tuple[bytes, str]:
    figure = figure or FigureOptions()
    if not layers:
        layers = []
    layers = [Layer(**L) if isinstance(L, dict) else L for L in layers]
    theme = dict(theme_of(figure.theme))
    if figure.background:
        theme["background"] = check_color(figure.background)
        if (pitch is None or not getattr(pitch, "pitch_color", None)) and theme["pitch_color"] != "grass":
            theme["pitch_color"] = theme["background"]
    chart_df = get_df(dataset_id, data, filters) if (dataset_id or data) else None
    layer_dfs = [_layer_df(L, chart_df) for L in layers]
    for L, ldf in zip(layers, layer_dfs):  # fix category order/colours across facets
        if L.color_by and ldf is not None and "order" not in L.options and L.color_by in ldf.columns \
                and not is_continuous(ldf[L.color_by]):
            L.options["order"] = ordered_categories(ldf[L.color_by])

    facets: list[Any] = [None]
    facet_col = None
    if figure.facet_by:
        src = chart_df if chart_df is not None else next((d for d in layer_dfs if d is not None), None)
        if src is None:
            raise ToolError("facet_by needs chart data.")
        facet_col = resolve_column(src, figure.facet_by, "facet_by")
        facets = ordered_categories(src[facet_col], figure.facet_order)
        if figure.facet_order is None:
            try:
                facets = sorted(facets)
            except TypeError:
                pass
        if len(facets) > figure.max_facets:
            facets = facets[: figure.max_facets]
        if not facets:
            raise ToolError("No values to facet by after filtering.")

    with RENDER_LOCK:
        p = make_pitch(pitch, theme)
        n = len(facets)
        ncols = figure.ncols or (1 if n == 1 else min(4, math.ceil(math.sqrt(n))))
        nrows = math.ceil(n / ncols)
        vertical = bool(pitch and (pitch.vertical if isinstance(pitch, PitchConfig) else pitch.get("vertical")))
        need_cbar = figure.colorbar
        fig, axs = p.grid(figheight=figure.figheight, nrows=nrows, ncols=ncols,
                          grid_height=0.76 if n == 1 else 0.72, title_height=0.08 if n == 1 else 0.09,
                          endnote_height=0.04, title_space=0.02 if n == 1 else 0.07, endnote_space=0.02,
                          space=0.08 if n > 1 else 0.05, axis=False,
                          grid_width=0.9 if need_cbar and vertical else 0.95)
        fig.set_facecolor(theme["background"])
        pitch_axes = np.atleast_1d(axs["pitch"]).ravel() if isinstance(axs, dict) else np.atleast_1d(axs).ravel()
        ctx = Ctx(pitch=p, theme=theme, fs=figure.font_scale * (1 if n == 1 else max(0.6, 1.4 / math.sqrt(n))))
        summary: list[str] = []
        for i, fv in enumerate(facets):
            ax = pitch_axes[i]
            for L, ldf in zip(layers, layer_dfs):
                d = ldf
                if fv is not None and d is not None and facet_col in d.columns:
                    d = d[d[facet_col] == fv]
                if d is not None:
                    d = d.reset_index(drop=True)
                    if d.empty:
                        if i == 0 and fv is None:
                            ctx.warnings.append(f"layer '{L.type}' has no rows after filters")
                        continue
                try:
                    msg = IMPLS[L.type](ctx, ax, d, L)
                except ToolError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise ToolError(f"Layer '{L.type}' failed: {type(exc).__name__}: {exc}") from exc
                summary.append(f"[{fv}] {msg}" if fv is not None else msg)
            if fv is not None:
                ax.set_title(str(fv), color=theme["text"], fontsize=14 * ctx.fs * 1.2, pad=6)
        for ax in pitch_axes[n:]:
            ax.remove()

        if figure.legend and ctx.handles:
            ax0 = pitch_axes[0]
            leg_kw = dict(facecolor=theme["background"], edgecolor=theme["muted"], labelcolor=theme["text"],
                          fontsize=11 * figure.font_scale, framealpha=0.9)
            if figure.legend_loc == "below":
                ax0 = pitch_axes[min(n, len(pitch_axes)) - 1] if n > 1 else ax0
                leg = ax0.legend(list(ctx.handles.values()), list(ctx.handles.keys()), loc="upper center",
                                 bbox_to_anchor=(0.5, 0.0), ncol=min(len(ctx.handles), 5), **leg_kw)
            else:
                leg = ax0.legend(list(ctx.handles.values()), list(ctx.handles.keys()), loc=figure.legend_loc,
                                 **leg_kw)
            leg.set_zorder(20)
        if figure.colorbar and ctx.mappables:
            mappable, label = ctx.mappables[-1]
            last = pitch_axes[n - 1]
            if vertical:
                cax = last.inset_axes([1.03, 0.15, 0.03, 0.7])
                cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
            else:
                cax = last.inset_axes([0.0, -0.05, 0.3, 0.025])
                cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            cb.set_label(figure.colorbar_label or label, color=theme["text"], fontsize=10 * figure.font_scale)
            cb.ax.tick_params(colors=theme["text"], labelsize=9 * figure.font_scale)
            cb.outline.set_edgecolor(theme["muted"])
            if "%" in str(label):
                from matplotlib.ticker import PercentFormatter
                cb.ax.xaxis.set_major_formatter(PercentFormatter(1.0)) if not vertical else \
                    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        add_texts(axs if isinstance(axs, dict) else {}, theme, figure.title, figure.subtitle, figure.endnote,
                  title_size=26 * figure.font_scale)
        png = fig_to_png(fig, figure.dpi)

    text = "; ".join(summary) if summary else "empty pitch"
    if ctx.notes:
        text += "\nNotes: " + "; ".join(dict.fromkeys(ctx.notes))
    if ctx.warnings:
        text += "\nWarnings: " + "; ".join(dict.fromkeys(ctx.warnings))
    return png, text
