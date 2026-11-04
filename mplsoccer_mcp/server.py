"""MCP server exposing mplsoccer charting as tools (remote, Streamable HTTP)."""

from __future__ import annotations

import os
import secrets
import threading
from collections import OrderedDict
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from mcp.server.mcpserver import Image, MCPServer
from mplsoccer import Pitch, Standardizer
from pydantic import Field

from . import data as D
from .data import STORE, Filter, ToolError
from .other_charts import render_bumpy, render_joint, render_pizza, render_radar
from .pitch_charts import FigureOptions, Layer, render_pitch_chart
from .reference import reference
from .render import PITCH_TYPES, PitchConfig, make_pitch, theme_of

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
MAX_CHARTS = int(os.getenv("MPLSOCCER_MCP_MAX_CHARTS", "300"))

INSTRUCTIONS = """\
Football (soccer) chart server built on the Python library mplsoccer. The user brings their own data
(event data, tracking freeze-frames, player stats, league tables). Typical flow:
load_dataset -> describe_dataset -> a chart tool (shot_map, pass_map, heatmap, pass_network, flow_map,
pass_sonar, freeze_frame, formation_chart, joint_plot, radar_chart, pizza_chart, bumpy_chart) or the fully
composable pitch_chart (stack any layers: scatter, arrows, comet lines, kde, hexbin, heatmaps, positional
zones, sonars, voronoi, convex hulls, annotations...). Small data can be passed inline as `data` records.
Always set pitch.pitch_type to match the data's coordinate system (describe_dataset suggests one).
Call get_reference for the full list of layer options. Charts are returned as PNG images with a text summary.
"""

mcp = MCPServer(name="mplsoccer", title="mplsoccer charts", instructions=INSTRUCTIONS, version="1.0.0")

# --------------------------------------------------------------------------- #
# chart cache (for download links)
# --------------------------------------------------------------------------- #

_CHARTS: "OrderedDict[str, bytes]" = OrderedDict()
_CHARTS_LOCK = threading.Lock()


def store_chart(png: bytes) -> str:
    cid = secrets.token_urlsafe(9)
    with _CHARTS_LOCK:
        _CHARTS[cid] = png
        while len(_CHARTS) > MAX_CHARTS:
            _CHARTS.popitem(last=False)
    if PUBLIC_BASE_URL:  # keep a copy so download links survive restarts / container sleep
        STORE.save_chart(cid, png)
    return cid


def get_chart(cid: str) -> bytes | None:
    with _CHARTS_LOCK:
        png = _CHARTS.get(cid)
    return png if png is not None else STORE.load_chart(cid)


def _result(png: bytes, summary: str) -> list[Any]:
    cid = store_chart(png)
    text = summary
    if PUBLIC_BASE_URL:
        text += f"\nDownload: {PUBLIC_BASE_URL}/charts/{cid}.png"
    return [Image(data=png, format="png"), text]


def _err(exc: Exception) -> ToolError:
    return exc if isinstance(exc, ToolError) else ToolError(f"{type(exc).__name__}: {exc}")


# Common annotated parameter types ------------------------------------------------
DatasetId = Annotated[str | None, Field(description="Id returned by load_dataset (or the dataset's name).")]
InlineData = Annotated[list[dict[str, Any]] | None,
                       Field(description="Inline records instead of a dataset_id, e.g. [{'x':100,'y':40,...}].")]
Filters = Annotated[list[Filter] | None, Field(description="Row filters, AND-combined, e.g. "
                                                           "[{'column':'team','op':'==','value':'Saprissa'}].")]
PitchArg = Annotated[PitchConfig | None, Field(description="Pitch look + coordinate system (pitch_type!).")]
Col = Annotated[str | None, Field(description="Column name (optional; auto-detected for coordinates).")]
Title = Annotated[str | None, Field(description="Chart title.")]
Subtitle = Annotated[str | None, Field(description="Chart subtitle.")]
Endnote = Annotated[str | None, Field(description="Small bottom-right note: data source, author, @handle.")]
Theme = Annotated[Literal["light", "dark", "grass", "night"], Field(description="Colour theme.")]
FacetBy = Annotated[str | None, Field(description="Draw one pitch per value of this column (small multiples).")]
ColorMap = Annotated[dict[str, str] | None, Field(description="Colours per category, e.g. {'Goal':'#D7263D'}.")]
Dpi = Annotated[int, Field(ge=50, le=300, description="Image resolution.")]


def _fig(title, subtitle, endnote, theme, facet_by=None, dpi=150, **kw) -> FigureOptions:
    return FigureOptions(title=title, subtitle=subtitle, endnote=endnote, theme=theme, facet_by=facet_by, dpi=dpi,
                         **{k: v for k, v in kw.items() if v is not None})


def _pitch(p: PitchConfig | None, **defaults) -> PitchConfig:
    if p is None:
        return PitchConfig(**defaults)
    return p


# =========================================================================== #
# DATA TOOLS
# =========================================================================== #


@mcp.tool(structured_output=False)
def load_dataset(
    name: Annotated[str, Field(description="Short name for the dataset, e.g. 'saprissa_vs_alajuelense'.")],
    csv_text: Annotated[str | None, Field(description="CSV/TSV text (separator auto-detected; ';' implies "
                                                      "decimal ',').")] = None,
    json_data: Annotated[Any, Field(description="JSON records (list of objects) or nested JSON such as "
                                                "StatsBomb events; nested fields are flattened with '.' and "
                                                "[x, y] lists become <col>_x/<col>_y.")] = None,
    url: Annotated[str | None, Field(description="Public http(s) URL to a CSV/JSON/Excel file, or a Google "
                                                 "Sheets link (shared as 'anyone with the link').")] = None,
    file_base64: Annotated[str | None, Field(description="Base64 file content (csv, xlsx, json, parquet).")] = None,
    file_name: Annotated[str | None, Field(description="Original file name, used to detect the format.")] = None,
    file_format: Annotated[Literal["csv", "json", "jsonl", "excel", "parquet"] | None,
                           Field(description="Force a format.")] = None,
    sheet: Annotated[str | int | None, Field(description="Excel sheet name or index.")] = None,
    separator: Annotated[str | None, Field(description="CSV separator if auto-detection fails.")] = None,
    decimal: Annotated[str | None, Field(description="Decimal mark ('.' or ',').")] = None,
) -> dict[str, Any]:
    """Load the user's own data (event data, shots, passes, tracking frames, player stats...) and return a
    dataset_id to use in chart tools, plus a description of its columns."""
    try:
        df, source = D.load_dataframe(csv_text=csv_text, json_data=json_data, url=url, file_base64=file_base64,
                                      file_name=file_name, file_format=file_format, sheet=sheet,
                                      separator=separator, decimal=decimal)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc
    ds = STORE.put(df, name, source)
    info = D.describe(df, max_unique=15, sample_rows=3)
    return {"dataset_id": ds.id, "name": ds.name, **info}


@mcp.tool(structured_output=False)
def list_datasets() -> list[dict[str, Any]]:
    """List datasets currently loaded on the server."""
    return [{"dataset_id": m["id"], "name": m.get("name"), "rows": m.get("rows"), "columns": m.get("columns"),
             "source": str(m.get("source", ""))[:120]} for m in STORE.list()]


@mcp.tool(structured_output=False)
def describe_dataset(
    dataset_id: Annotated[str, Field(description="Dataset id or name.")],
    columns: Annotated[list[str] | None, Field(description="Only describe these columns.")] = None,
    filters: Filters = None,
    max_unique: Annotated[int, Field(ge=1, le=200, description="Show value counts for text columns with up "
                                                               "to this many distinct values.")] = 25,
    sample_rows: Annotated[int, Field(ge=0, le=50)] = 5,
) -> dict[str, Any]:
    """Columns, types, ranges, category values, sample rows and a suggested pitch_type for a dataset."""
    try:
        df = D.apply_filters(STORE.get(dataset_id).df, filters)
        if columns:
            df = df[[D.resolve_column(df, c) for c in columns]]
        return D.describe(df, max_unique=max_unique, sample_rows=sample_rows)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def transform_dataset(
    dataset_id: Annotated[str, Field(description="Source dataset id.")],
    new_name: Annotated[str, Field(description="Name for the resulting dataset.")],
    filters: Filters = None,
    keep_columns: Annotated[list[str] | None, Field(description="Keep only these columns.")] = None,
    rename: Annotated[dict[str, str] | None, Field(description="Rename columns {old: new}.")] = None,
    standardize_from: Annotated[str | None, Field(description="Convert coordinates FROM this pitch_type...")] = None,
    standardize_to: Annotated[str | None, Field(description="...TO this pitch_type (e.g. opta -> statsbomb).")] = None,
    coordinate_pairs: Annotated[list[list[str]] | None,
                                Field(description="Coordinate column pairs to convert, e.g. "
                                                  "[['x','y'],['end_x','end_y']] (auto-detected if omitted).")] = None,
    length_from: float | None = None, width_from: float | None = None,
    length_to: float | None = None, width_to: float | None = None,
    add_angle_distance: Annotated[bool, Field(description="Add 'angle' (radians) and 'distance' columns from "
                                                          "start->end coordinates.")] = False,
    pitch_type: Annotated[str, Field(description="Pitch type used for add_angle_distance.")] = "statsbomb",
    sort_by: Annotated[list[str] | None, Field(description="Sort by these columns.")] = None,
    descending: bool = False,
    head: Annotated[int | None, Field(description="Keep only the first N rows (after sorting).")] = None,
) -> dict[str, Any]:
    """Create a new dataset by filtering, selecting/renaming columns, converting coordinates between pitch
    types (mplsoccer Standardizer), computing pass angles/distances, sorting or taking the top N rows."""
    try:
        df = D.apply_filters(STORE.get(dataset_id).df, filters).copy()
        if rename:
            df = df.rename(columns={D.resolve_column(df, k): v for k, v in rename.items()})
        if standardize_from and standardize_to:
            st = Standardizer(pitch_from=standardize_from, pitch_to=standardize_to, length_from=length_from,
                              width_from=width_from, length_to=length_to, width_to=width_to)
            pairs = coordinate_pairs
            if not pairs:
                pairs = []
                for a, b in (("x", "y"), ("end_x", "end_y")):
                    ca, cb = D.auto_column(df, None, a, False), D.auto_column(df, None, b, False)
                    if ca and cb:
                        pairs.append([ca, cb])
            if not pairs:
                raise ToolError("No coordinate columns found; pass coordinate_pairs.")
            for cx, cy in pairs:
                cx, cy = D.resolve_column(df, cx), D.resolve_column(df, cy)
                xs, ys = D.numeric(df, cx), D.numeric(df, cy)
                ok = np.isfinite(xs) & np.isfinite(ys)
                nx, ny = xs.copy(), ys.copy()
                if ok.any():
                    tx, ty = st.transform(xs[ok], ys[ok])
                    nx[ok], ny[ok] = tx, ty
                df[cx], df[cy] = nx, ny
        if add_angle_distance:
            kw = {}
            if pitch_type in ("custom", "tracab", "skillcorner", "secondspectrum", "metricasports"):
                kw = dict(pitch_length=length_to or 105, pitch_width=width_to or 68)
            p = Pitch(pitch_type=pitch_type, **kw)
            cx, cy = D.auto_column(df, None, "x"), D.auto_column(df, None, "y")
            ex, ey = D.auto_column(df, None, "end_x"), D.auto_column(df, None, "end_y")
            ang, dist = p.calculate_angle_and_distance(D.numeric(df, cx), D.numeric(df, cy),
                                                       D.numeric(df, ex), D.numeric(df, ey))
            df["angle"], df["distance"] = ang, dist
        if sort_by:
            df = df.sort_values([D.resolve_column(df, c) for c in sort_by], ascending=not descending)
        if head:
            df = df.head(head)
        if keep_columns:
            df = df[[D.resolve_column(df, c) for c in keep_columns]]
        ds = STORE.put(df.reset_index(drop=True), new_name, f"transform:{dataset_id}")
        return {"dataset_id": ds.id, "name": ds.name, **D.describe(df, max_unique=10, sample_rows=3)}
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def delete_dataset(dataset_id: Annotated[str, Field(description="Dataset id.")]) -> str:
    """Delete a dataset from the server."""
    return "deleted" if STORE.delete(dataset_id) else "not found"


@mcp.tool(structured_output=False)
def get_reference(
    topic: Annotated[Literal["all", "workflow", "pitch", "layers", "other"],
                     Field(description="workflow, pitch (types/options), layers (pitch_chart), other charts.")] = "all",
) -> str:
    """Full documentation of pitch types, themes, every pitch_chart layer option and the other chart tools."""
    return reference(topic)


@mcp.tool(structured_output=False)
def list_formations() -> dict[str, Any]:
    """Valid formations for formation_chart / formation layers, with their position names."""
    p = Pitch()
    return {f: [pos.name for pos in p.get_formation(f)] for f in p.formations}


# =========================================================================== #
# GENERIC PITCH TOOLS
# =========================================================================== #


@mcp.tool(structured_output=False)
def draw_pitch(
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    figheight: Annotated[float, Field(ge=3, le=24)] = 9, dpi: Dpi = 150,
) -> list[Any]:
    """Draw an empty pitch (any pitch type, vertical/half, grass, stripes, Juego de Posición lines...).
    Useful to preview styles or as a template."""
    try:
        png, s = render_pitch_chart(pitch=pitch, layers=[], figure=FigureOptions(
            title=title, subtitle=subtitle, endnote=endnote, theme=theme, figheight=figheight, dpi=dpi))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def pitch_chart(
    layers: Annotated[list[Layer], Field(description="Ordered layers (first is drawn at the bottom). See "
                                                     "get_reference(topic='layers').")],
    dataset_id: DatasetId = None,
    data: InlineData = None,
    filters: Filters = None,
    pitch: PitchArg = None,
    figure: Annotated[FigureOptions | None, Field(description="Title, theme, legend, colorbar, facets, size.")] = None,
) -> list[Any]:
    """The fully composable chart: stack any mplsoccer pitch layers — scatter, arrows, lines/comets, plot paths,
    kde, hexbin, grid/positional/zone heatmaps with labels, flow, sonars, pass network, voronoi, delaunay,
    convex hulls, goal angles, text, annotations, polygons, formations — with optional small multiples
    (figure.facet_by). Layers use the chart dataset unless they set their own dataset_id/data, and each
    layer can have its own filters (e.g. grey misses + red goals)."""
    try:
        png, s = render_pitch_chart(pitch=pitch, layers=layers, dataset_id=dataset_id, data=data,
                                    filters=filters, figure=figure)
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


# =========================================================================== #
# HIGH-LEVEL PITCH TOOLS
# =========================================================================== #


@mcp.tool(structured_output=False)
def shot_map(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None,
    xg: Annotated[str | None, Field(description="xG column: marker size proportional to xG.")] = None,
    outcome: Annotated[str | None, Field(description="Outcome column used for colour (Goal/Saved/Off T...).")] = None,
    goal_value: Annotated[str, Field(description="Value of the outcome column meaning a goal.")] = "Goal",
    color_map: ColorMap = None,
    goals_as_football: Annotated[bool, Field(description="Draw goals with a football marker on top.")] = True,
    label_by: Annotated[str | None, Field(description="Write this column (e.g. player, minute) by each shot.")] = None,
    show_goal_angle: Annotated[bool, Field(description="Shade the shooting angle (best for few shots).")] = False,
    show_kde: Annotated[bool, Field(description="Add a density layer of shot locations underneath.")] = False,
    pitch: Annotated[PitchConfig | None, Field(description="Default: vertical half pitch, statsbomb.")] = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    facet_by: FacetBy = None, dpi: Dpi = 150,
) -> list[Any]:
    """Shot map: shot locations sized by xG, coloured by outcome, goals as footballs, optional goal angles,
    density layer and one pitch per player/team (facet_by). Summary includes shots, goals and total xG."""
    try:
        p = _pitch(pitch, vertical=True, half=True)
        df = D.get_df(dataset_id, data, filters)
        layers: list[Layer] = []
        if show_kde:
            layers.append(Layer(type="kde", x=x, y=y, alpha=0.6))
        if show_goal_angle:
            layers.append(Layer(type="goal_angle", x=x, y=y, alpha=0.15))
        cmap = dict(color_map or {})
        oc = D.resolve_column(df, outcome, "outcome") if outcome else None
        if oc:
            neutral = ["#2E86DE", "#9AA0A6", "#F49D37", "#7D4E9E", "#1B998B", "#8C564B", "#17BECF", "#BC5090"]
            cats = [c for c in pd.Series(df[oc]).dropna().value_counts().index]
            low = {str(k).lower() for k in cmap}
            i = 0
            for c in cats:
                if str(c).lower() == goal_value.lower():
                    cmap.setdefault(str(c), "#D7263D")
                elif str(c).lower() not in low:
                    cmap[str(c)] = neutral[i % len(neutral)]
                    i += 1
        goal_filter = [Filter(column=oc, op="==", value=goal_value)] if oc else None
        base_filters = [Filter(column=oc, op="!=", value=goal_value)] if (oc and goals_as_football) else None
        layers.append(Layer(type="scatter", x=x, y=y, size_by=xg, color_by=oc, color_map=cmap or None,
                            filters=base_filters, label_by=label_by, alpha=0.85, size=220,
                            options={"size_range": [60, 1200], "size_max": 1.0} if xg else {}))
        if oc and goals_as_football:
            layers.append(Layer(type="scatter", x=x, y=y, size_by=xg, filters=goal_filter, marker="football",
                                label_by=label_by, size=300, zorder=5, label=goal_value, color="#111111",
                                options={"size_range": [120, 1300], "size_max": 1.0} if xg else {}))
        png, s = render_pitch_chart(pitch=p, layers=layers, dataset_id=dataset_id, data=data, filters=filters,
                                    figure=_fig(title, subtitle, endnote, theme, facet_by, dpi))
        stats = f"shots={len(df)}"
        if oc:
            stats += f", goals={int((df[oc].astype(str).str.lower() == goal_value.lower()).sum())}"
        if xg:
            stats += f", xG={pd.to_numeric(df[D.resolve_column(df, xg)], errors='coerce').sum():.2f}"
        if oc and goals_as_football:
            s += "\nGoals drawn as football markers."
        return _result(png, stats + "\n" + s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def pass_map(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None, end_x: Col = None, end_y: Col = None,
    color_by: Annotated[str | None, Field(description="Column to colour by (e.g. outcome, player, "
                                                      "pass type, or a numeric column such as xT).")] = None,
    color_map: ColorMap = None,
    style: Annotated[Literal["arrows", "lines", "comet"], Field(description="Arrow, line or tapered comet.")] = "arrows",
    color: Annotated[str | None, Field(description="Single colour when color_by is not used.")] = None,
    background: Annotated[Literal["none", "kde", "heatmap", "positional"],
                          Field(description="Density of pass origins drawn underneath.")] = "none",
    width: Annotated[float | None, Field(description="Arrow width / line width.")] = None,
    show_start: Annotated[bool, Field(description="Dot at the start of each pass.")] = False,
    label_by: Col = None,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    facet_by: FacetBy = None, legend_loc: Annotated[str, Field(description="'upper left', 'below', ...")] = "upper left",
    dpi: Dpi = 150,
) -> list[Any]:
    """Pass (or carry / any start->end action) map with arrows, lines or comets, coloured by outcome/type/
    player or a numeric value, optional density background and small multiples per player (facet_by)."""
    try:
        layers: list[Layer] = []
        if background == "kde":
            layers.append(Layer(type="kde", x=x, y=y, alpha=0.5))
        elif background == "heatmap":
            layers.append(Layer(type="heatmap", x=x, y=y, alpha=0.6, options={"bins": [6, 5]}))
        elif background == "positional":
            layers.append(Layer(type="heatmap_positional", x=x, y=y, alpha=0.6))
        layers.append(Layer(type=style, x=x, y=y, end_x=end_x, end_y=end_y, color_by=color_by, color_map=color_map,
                            color=color, size=width, label_by=label_by, options={"show_start": show_start}))
        fig = _fig(title, subtitle, endnote, theme, facet_by, dpi, legend_loc=legend_loc,
                   colorbar=background == "none" or color_by is not None)
        png, s = render_pitch_chart(pitch=pitch, layers=layers, dataset_id=dataset_id, data=data, filters=filters,
                                    figure=fig)
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def heatmap(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None,
    kind: Annotated[Literal["grid", "positional", "zones", "kde", "hexbin"],
                    Field(description="grid = bin_statistic rectangles; positional = Juego de Posición zones; "
                                      "zones = custom rectangles; kde = smooth density; hexbin = hexagons.")] = "grid",
    bins: Annotated[list[int] | None, Field(description="[nx, ny] for grid heatmaps, e.g. [6, 5] or [3, 3] "
                                                        "(thirds), [18, 12] for fine grids.")] = None,
    positional: Annotated[Literal["full", "horizontal", "vertical"], Field(description="For kind=positional.")] = "full",
    zones: Annotated[list[list[float]] | None, Field(description="For kind=zones: [[x0,x1,y0,y1], ...] tiling "
                                                                 "the pitch.")] = None,
    zone_names: list[str] | None = None,
    statistic: Annotated[Literal["count", "mean", "sum", "median", "std", "min", "max"],
                         Field(description="Aggregation; non-count needs `values`.")] = "count",
    values: Annotated[str | None, Field(description="Numeric column to aggregate (e.g. xT, xG, success).")] = None,
    normalize: Annotated[bool, Field(description="Show share of total instead of counts.")] = False,
    show_labels: Annotated[bool, Field(description="Write the value in each bin/zone.")] = False,
    label_format: Annotated[str | None, Field(description="e.g. '{:.0%}', '{:.0f}', '{:.2f}'.")] = None,
    smooth: Annotated[float | None, Field(description="Gaussian smoothing sigma for grid heatmaps (e.g. 1).")] = None,
    cmap: Annotated[str | None, Field(description="Matplotlib colormap (Reds, hot, viridis, YlOrRd, ...).")] = None,
    overlay_points: Annotated[bool, Field(description="Also draw the raw points.")] = False,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    facet_by: FacetBy = None, dpi: Dpi = 150,
) -> list[Any]:
    """Heatmaps of where actions happen (touches, defensive actions, receptions, shots...): grid bins,
    Juego de Posición zones, custom zones, KDE or hexbin; counts, percentages or any statistic of a value."""
    try:
        o: dict[str, Any] = {"statistic": statistic, "normalize": normalize, "show_labels": show_labels}
        if label_format:
            o["label_format"] = label_format
        if kind == "grid":
            o["bins"] = bins or [6, 5]
            if smooth:
                o["smooth"] = smooth
            layer = Layer(type="heatmap", x=x, y=y, values=values, cmap=cmap, options=o)
        elif kind == "positional":
            o["positional"] = positional
            layer = Layer(type="heatmap_positional", x=x, y=y, values=values, cmap=cmap, options=o)
        elif kind == "zones":
            o["zones"] = zones or "positional_full"
            if zone_names:
                o["names"] = zone_names
            layer = Layer(type="heatmap_zones", x=x, y=y, values=values, cmap=cmap, options=o)
        elif kind == "kde":
            layer = Layer(type="kde", x=x, y=y, cmap=cmap)
        else:
            layer = Layer(type="hexbin", x=x, y=y, cmap=cmap, values=values,
                          options={"statistic": statistic if statistic != "count" else "mean",
                                   **({"gridsize": bins} if bins else {})})
        layers = [layer]
        if overlay_points:
            layers.append(Layer(type="scatter", x=x, y=y, size=25, alpha=0.6, color="#222222"))
        png, s = render_pitch_chart(pitch=pitch, layers=layers, dataset_id=dataset_id, data=data, filters=filters,
                                    figure=_fig(title, subtitle, endnote, theme, facet_by, dpi))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def flow_map(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None, end_x: Col = None, end_y: Col = None,
    bins: Annotated[list[int] | None, Field(description="[nx, ny] grid, default [6, 4].")] = None,
    arrow_type: Annotated[Literal["average", "same", "scale"],
                          Field(description="scale = length scaled by mean distance (max = arrow_length); "
                                            "same = fixed length; average = true mean distance.")] = "scale",
    arrow_length: float = 15,
    background_heatmap: Annotated[bool, Field(description="Heatmap of action origins under the arrows.")] = True,
    cmap: Annotated[str | None, Field(description="Heatmap colormap.")] = None,
    arrow_color: str | None = None,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    facet_by: FacetBy = None, dpi: Dpi = 150,
) -> list[Any]:
    """Flow map: average direction of passes/carries from each zone, optionally over a heatmap of origins."""
    try:
        b = bins or [6, 4]
        layers = []
        if background_heatmap:
            layers.append(Layer(type="heatmap", x=x, y=y, cmap=cmap or "Blues", alpha=0.8, options={"bins": b}))
        layers.append(Layer(type="flow", x=x, y=y, end_x=end_x, end_y=end_y, color=arrow_color,
                            options={"bins": b, "arrow_type": arrow_type, "arrow_length": arrow_length}))
        png, s = render_pitch_chart(pitch=pitch, layers=layers, dataset_id=dataset_id, data=data, filters=filters,
                                    figure=_fig(title, subtitle, endnote, theme, facet_by, dpi))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def pass_network(
    passer: Annotated[str, Field(description="Column with the player who passes.")],
    recipient: Annotated[str, Field(description="Column with the player who receives.")],
    dataset_id: DatasetId = None, data: InlineData = None,
    filters: Annotated[list[Filter] | None, Field(description="Usually: one team, successful passes, before the "
                                                              "first substitution.")] = None,
    x: Col = None, y: Col = None, end_x: Col = None, end_y: Col = None,
    min_passes: Annotated[int, Field(ge=1, description="Minimum passes between a pair to draw a line.")] = 3,
    players: Annotated[list[str] | None, Field(description="Only include these players (e.g. the starting XI).")] = None,
    label: Annotated[Literal["name", "last_name", "initials", "none"], Field(description="Node labels.")] = "last_name",
    labels: Annotated[dict[str, str] | None, Field(description="Custom labels {player: 'label' or shirt number}.")] = None,
    node_color: str | None = None, edge_color: str | None = None,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "dark",
    facet_by: Annotated[str | None, Field(description="E.g. team or half for side-by-side networks.")] = None,
    dpi: Dpi = 150,
) -> list[Any]:
    """Passing network: average positions (node size = passes made) and connections (line width = passes
    between the pair). The summary lists the most frequent combinations."""
    try:
        o: dict[str, Any] = {"recipient": recipient, "min_passes": min_passes, "label": label}
        if players:
            o["players"] = players
        if labels:
            o["labels"] = labels
            o["label_inside"] = all(len(str(v)) <= 3 for v in labels.values())
        if node_color:
            o["node_color"] = node_color
        if edge_color:
            o["edge_color"] = edge_color
        layer = Layer(type="pass_network", group_by=passer, x=x, y=y, end_x=end_x, end_y=end_y, options=o)
        png, s = render_pitch_chart(pitch=pitch, layers=[layer], dataset_id=dataset_id, data=data, filters=filters,
                                    figure=_fig(title, subtitle, endnote, theme, facet_by, dpi))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def pass_sonar(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None, end_x: Col = None, end_y: Col = None,
    angle_column: Annotated[str | None, Field(description="Use an existing angle column instead of end coords.")] = None,
    angle_degrees: Annotated[bool, Field(description="angle_column is in degrees.")] = False,
    bins: Annotated[list[int] | None, Field(description="[nx, ny] grid of sonars (default [6, 4]; [1, 1] = "
                                                        "one big sonar).")] = None,
    angle_bins: Annotated[int, Field(ge=4, le=36, description="Number of direction slices.")] = 10,
    positional_zones: Annotated[bool, Field(description="One sonar per Juego de Posición zone instead of a grid.")] = False,
    color_by: Annotated[Literal["distance", "none"], Field(description="Colour slices by mean pass distance.")] = "distance",
    cmap: str | None = None,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    facet_by: FacetBy = None, dpi: Dpi = 150,
) -> list[Any]:
    """Pass sonars: polar bar charts showing how often (length) and how far (colour) passes go in each
    direction, per pitch zone."""
    try:
        o: dict[str, Any] = {"angle_bins": angle_bins, "color_by_stat": color_by, "bins": bins or [6, 4]}
        if angle_column:
            o["angle_column"] = angle_column
            o["angle_degrees"] = angle_degrees
            if color_by == "distance":
                o["color_by_stat"] = "none"
        if positional_zones:
            o["zones"] = "positional_full"
        layers = []
        if positional_zones:
            layers.append(Layer(type="zones", options={"zones": "positional_full", "show_names": False}, alpha=0.5))
        layers.append(Layer(type="sonar", x=x, y=y, end_x=end_x, end_y=end_y, cmap=cmap, options=o))
        png, s = render_pitch_chart(pitch=pitch, layers=layers, dataset_id=dataset_id, data=data, filters=filters,
                                    figure=_fig(title, subtitle, endnote, theme, facet_by, dpi))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def freeze_frame(
    team: Annotated[str, Field(description="Column telling which team each player belongs to "
                                           "(or a teammate True/False column).")],
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None,
    team_a: Annotated[str | None, Field(description="Value of `team` for the attacking/first team.")] = None,
    color_map: ColorMap = None,
    show_voronoi: Annotated[bool, Field(description="Space controlled by each player.")] = True,
    show_delaunay: Annotated[bool, Field(description="Delaunay triangulation per team.")] = False,
    show_hull: Annotated[bool, Field(description="Convex hull (team shape) per team.")] = False,
    label_by: Annotated[str | None, Field(description="Label players (name / number column).")] = None,
    ball: Annotated[list[float] | None, Field(description="[x, y] of the ball / shooter.")] = None,
    shot_angle: Annotated[bool, Field(description="Shade the goal angle from the ball position.")] = False,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    facet_by: Annotated[str | None, Field(description="E.g. a frame/event id column for several frames.")] = None,
    dpi: Dpi = 150,
) -> list[Any]:
    """Tracking / freeze-frame snapshot: players of both teams with Voronoi control areas, Delaunay
    triangulation, team convex hulls, ball position and shooting angle."""
    try:
        layers = []
        if show_voronoi:
            layers.append(Layer(type="voronoi", x=x, y=y, group_by=team, color_map=color_map, alpha=0.25,
                                options={"team_a": team_a, "show_players": False}))
        if show_hull:
            layers.append(Layer(type="convexhull", x=x, y=y, group_by=team, color_map=color_map, alpha=0.15,
                                options={"show_points": False}))
        if show_delaunay:
            layers.append(Layer(type="delaunay", x=x, y=y, group_by=team, color_map=color_map, alpha=0.6))
        if ball and shot_angle:
            layers.append(Layer(type="goal_angle", data=[{"x": ball[0], "y": ball[1]}], x="x", y="y", alpha=0.2))
        order = [team_a] if team_a else None
        layers.append(Layer(type="scatter", x=x, y=y, color_by=team, color_map=color_map, size=260,
                            label_by=label_by, options={"edgecolor": "#111111", "order": order}))
        if ball:
            layers.append(Layer(type="scatter", data=[{"x": ball[0], "y": ball[1]}], x="x", y="y",
                                marker="football", size=220, zorder=6))
        png, s = render_pitch_chart(pitch=pitch, layers=layers, dataset_id=dataset_id, data=data, filters=filters,
                                    figure=_fig(title, subtitle, endnote, theme, facet_by, dpi))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def formation_chart(
    formation: Annotated[str, Field(description="e.g. '4-3-3', '4-2-3-1', '3-5-2' (see list_formations).")],
    players: Annotated[dict[str, str] | list[str] | None,
                       Field(description="Names per position {'GK':'Navas','ST':'Campbell'} or a list in the "
                                         "formation's position order (see list_formations).")] = None,
    numbers: Annotated[dict[str, int] | list[int] | None, Field(description="Shirt numbers per position or list.")] = None,
    color: str | None = None,
    opponent_formation: Annotated[str | None, Field(description="Second team's formation (drawn attacking "
                                                                "the other way).")] = None,
    opponent_players: dict[str, str] | list[str] | None = None,
    opponent_numbers: dict[str, int] | list[int] | None = None,
    opponent_color: str | None = None,
    team_name: str | None = None, opponent_name: str | None = None,
    marker: Annotated[str | None, Field(description="'o' (default), 'shirt', 's', 'h'...")] = None,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "grass",
    dpi: Dpi = 150,
) -> list[Any]:
    """Line-up / formation graphic for one team, or two teams facing each other."""
    try:
        th = theme_of(theme)
        two = bool(opponent_formation)
        layers = [Layer(type="formation", color=color or th["palette"][0], marker=marker, label=team_name,
                        options={"formation": formation, "players": players, "numbers": numbers,
                                 "half": two, "size": None})]
        if two:
            layers.append(Layer(type="formation", color=opponent_color or th["palette"][1], marker=marker,
                                label=opponent_name,
                                options={"formation": opponent_formation, "players": opponent_players,
                                         "numbers": opponent_numbers, "half": True, "flip": True}))
        p = pitch or PitchConfig(pitch_type="statsbomb", vertical=not two)
        if p.pitch_type != "statsbomb" and p.pitch_type not in ("opta", "wyscout", "uefa"):
            p = p.model_copy(update={"pitch_type": "statsbomb"})
        png, s = render_pitch_chart(pitch=p, layers=layers, figure=_fig(title, subtitle, endnote, theme, None, dpi,
                                                                         legend=bool(team_name or opponent_name)))
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def joint_plot(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None,
    kind: Annotated[Literal["kde", "scatter", "hexbin"], Field(description="Main plot on the pitch.")] = "kde",
    marginal: Annotated[Literal["hist", "kde"], Field(description="Marginal distributions.")] = "hist",
    color: str | None = None, cmap: str | None = None,
    pitch: PitchArg = None,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    dpi: Dpi = 150,
) -> list[Any]:
    """Pitch with marginal distributions along the length and width (where along/across the pitch the
    actions happen)."""
    try:
        png, s = render_joint(dataset_id=dataset_id, data=data, filters=filters, x=x, y=y, pitch=pitch, kind=kind,
                              marginal=marginal, color=color, cmap=cmap, theme=theme, title=title,
                              subtitle=subtitle, endnote=endnote, dpi=dpi)
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def zone_stats(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x: Col = None, y: Col = None,
    kind: Annotated[Literal["grid", "positional"], Field(description="Grid bins or Juego de Posición zones.")] = "grid",
    bins: Annotated[list[int] | None, Field(description="[nx, ny], default [3, 3] (thirds x channels).")] = None,
    statistic: Literal["count", "mean", "sum", "median", "std", "min", "max"] = "count",
    values: Col = None,
    pitch_type: str = "statsbomb",
    pitch_length: float | None = None, pitch_width: float | None = None,
    group_by: Annotated[str | None, Field(description="Compute separately per value (e.g. team/player).")] = None,
) -> dict[str, Any]:
    """Numbers only (no image): counts/percentages or a statistic per pitch zone, to answer questions in text
    (e.g. 'what % of our touches are in the final third?'). Rows = along the pitch (own goal -> opponent goal)."""
    try:
        df = D.get_df(dataset_id, data, filters)
        p = make_pitch(PitchConfig(pitch_type=pitch_type, pitch_length=pitch_length, pitch_width=pitch_width),
                       theme_of("light"))
        xc, yc = D.auto_column(df, x, "x"), D.auto_column(df, y, "y")
        groups = [("all", df)]
        if group_by:
            gc = D.resolve_column(df, group_by, "group_by")
            groups = [(str(k), g) for k, g in df.groupby(gc)]
        vc = D.resolve_column(df, values, "values") if values else None
        out: dict[str, Any] = {}
        for gname, g in groups:
            xs, ys = D.numeric(g, xc), D.numeric(g, yc)
            ok = np.isfinite(xs) & np.isfinite(ys)
            vals = D.numeric(g, vc)[ok] if vc else None
            if statistic != "count" and vals is None:
                raise ToolError("statistic other than count needs `values`.")
            if kind == "grid":
                b = tuple(bins or [3, 3])
                st = p.bin_statistic(xs[ok], ys[ok], values=vals, statistic=statistic, bins=b)
                arr = np.asarray(st["statistic"], dtype=float)  # shape (ny, nx)
                cnt = p.bin_statistic(xs[ok], ys[ok], statistic="count", bins=b)["statistic"]
                total = float(np.nansum(cnt))
                cells = []
                ny, nx = arr.shape
                ystart = st["y_grid"][:, 0]
                for j in range(ny):
                    for i in range(nx):
                        cells.append({
                            "x_band": f"{i + 1}/{nx}", "y_band": f"{j + 1}/{ny}",
                            "x_range": [round(float(st["x_grid"][j, i]), 2), round(float(st["x_grid"][j, i + 1]), 2)],
                            "y_range": [round(float(ystart[j]), 2), round(float(st["y_grid"][j + 1, 0]), 2)],
                            "value": None if np.isnan(arr[j, i]) else round(float(arr[j, i]), 4),
                            "count": int(cnt[j, i]),
                            "share": round(float(cnt[j, i]) / total, 4) if total else 0,
                        })
                out[gname] = {"n": int(ok.sum()), "statistic": statistic, "cells": cells,
                              "share_by_x_band": [round(float(np.nansum(cnt[:, i])) / total, 4) if total else 0
                                                  for i in range(nx)]}
            else:
                st = p.bin_statistic_positional(xs[ok], ys[ok], values=vals, statistic=statistic, positional="full")
                cnt = p.bin_statistic_positional(xs[ok], ys[ok], statistic="count", positional="full")
                total = float(sum(np.nansum(c["statistic"]) for c in cnt))
                zones = []
                for k, (s1, c1) in enumerate(zip(st, cnt)):
                    for v, c in zip(np.ravel(s1["statistic"]), np.ravel(c1["statistic"])):
                        zones.append({"zone": len(zones) + 1, "value": None if np.isnan(v) else round(float(v), 4),
                                      "count": int(c), "share": round(float(c) / total, 4) if total else 0})
                out[gname] = {"n": int(ok.sum()), "statistic": statistic, "zones": zones}
        return out
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


# =========================================================================== #
# NON-PITCH CHARTS
# =========================================================================== #


@mcp.tool(structured_output=False)
def radar_chart(
    params: Annotated[list[str] | None, Field(description="Metric names (or stat column names in dataset mode).")] = None,
    values: Annotated[list[float] | None, Field(description="Player values (inline mode).")] = None,
    compare_values: Annotated[list[float] | None, Field(description="Second player's values.")] = None,
    min_range: Annotated[list[float] | None, Field(description="Lower bound per param (auto in dataset mode).")] = None,
    max_range: Annotated[list[float] | None, Field(description="Upper bound per param (auto in dataset mode).")] = None,
    lower_is_better: Annotated[list[str] | None, Field(description="Params where lower is better (fouls, "
                                                                   "turnovers...).")] = None,
    dataset_id: DatasetId = None, data: InlineData = None,
    filters: Annotated[list[Filter] | None, Field(description="Comparison pool for the auto ranges "
                                                              "(e.g. same position, >900 minutes).")] = None,
    name_column: Annotated[str | None, Field(description="Player-name column (dataset mode).")] = None,
    player: Annotated[str | None, Field(description="Player to plot (dataset mode).")] = None,
    compare_player: Annotated[str | None, Field(description="Second player (dataset mode).")] = None,
    player_label: str | None = None, compare_label: str | None = None,
    param_labels: Annotated[list[str] | None, Field(description="Display names for params.")] = None,
    style: Annotated[Literal["standard", "solid", "turbine"], Field(description="turbine needs a dataset.")] = "standard",
    color: str | None = None, compare_color: str | None = None,
    num_rings: Annotated[int, Field(ge=2, le=10)] = 4,
    round_int: bool | None = None,
    curved_labels: bool = False,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    dpi: Dpi = 150,
) -> list[Any]:
    """Player radar (mplsoccer Radar): one player or a comparison, standard/solid/turbine styles. In dataset
    mode the ranges come from the 5th-95th percentile of the (filtered) comparison pool."""
    try:
        png, s = render_radar(params=params, values=values, compare_values=compare_values, min_range=min_range,
                              max_range=max_range, lower_is_better=lower_is_better, dataset_id=dataset_id,
                              data=data, filters=filters, name_column=name_column, player=player,
                              compare_player=compare_player, player_label=player_label,
                              compare_label=compare_label, style=style, theme=theme, color=color,
                              compare_color=compare_color, title=title, subtitle=subtitle, endnote=endnote,
                              param_labels=param_labels, num_rings=num_rings, round_int=round_int,
                              curved_labels=curved_labels, dpi=dpi)
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def pizza_chart(
    params: Annotated[list[str] | None, Field(description="Metric names (or stat columns in dataset mode).")] = None,
    values: Annotated[list[float] | None, Field(description="Values, usually percentiles 0-100 (inline mode).")] = None,
    compare_values: list[float] | None = None,
    min_range: list[float] | None = None, max_range: list[float] | None = None,
    dataset_id: DatasetId = None, data: InlineData = None,
    filters: Annotated[list[Filter] | None, Field(description="Comparison pool for percentiles.")] = None,
    name_column: str | None = None, player: str | None = None, compare_player: str | None = None,
    as_percentiles: Annotated[bool, Field(description="Dataset mode: convert to percentile ranks in the pool.")] = True,
    lower_is_better: list[str] | None = None,
    show_raw_values: Annotated[bool, Field(description="Print raw stat values on the slices.")] = False,
    param_labels: list[str] | None = None,
    param_groups: Annotated[dict[str, list[str]] | None,
                            Field(description="Colour slices by group, e.g. {'Ataque':[...], 'Defensa':[...]}.")] = None,
    group_colors: dict[str, str] | None = None,
    slice_color: str | None = None, compare_color: str | None = None,
    curved_labels: bool = False,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "light",
    dpi: Dpi = 150,
) -> list[Any]:
    """Percentile pizza chart (mplsoccer PyPizza), optionally grouped by category colours or comparing two
    players. Dataset mode computes percentiles against the filtered pool."""
    try:
        png, s = render_pizza(params=params, values=values, compare_values=compare_values, min_range=min_range,
                              max_range=max_range, dataset_id=dataset_id, data=data, filters=filters,
                              name_column=name_column, player=player, compare_player=compare_player,
                              as_percentiles=as_percentiles, lower_is_better=lower_is_better,
                              show_raw_values=show_raw_values, param_labels=param_labels,
                              param_groups=param_groups, group_colors=group_colors, slice_color=slice_color,
                              compare_color=compare_color, theme=theme, title=title, subtitle=subtitle,
                              endnote=endnote, curved_labels=curved_labels, dpi=dpi)
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc


@mcp.tool(structured_output=False)
def bumpy_chart(
    dataset_id: DatasetId = None, data: InlineData = None, filters: Filters = None,
    x_column: Annotated[str | None, Field(description="Round / matchday / season column.")] = None,
    item_column: Annotated[str | None, Field(description="Team or player column.")] = None,
    rank_column: Annotated[str | None, Field(description="League position column.")] = None,
    value_column: Annotated[str | None, Field(description="Or a value to rank by within each round (points).")] = None,
    ascending: Annotated[bool, Field(description="With value_column: lower value = better rank.")] = False,
    x_labels: Annotated[list[str] | None, Field(description="Inline mode: round labels.")] = None,
    ranks: Annotated[dict[str, list[int]] | None, Field(description="Inline mode: {team: [rank per round]}.")] = None,
    highlight: Annotated[dict[str, str | None] | None, Field(description="{team: colour} to highlight "
                                                                         "(default: final top 3).")] = None,
    x_label: str | None = None, y_label: str | None = None,
    rank_1_on_top: Annotated[bool, Field(description="Draw position 1 at the top.")] = True,
    title: Title = None, subtitle: Subtitle = None, endnote: Endnote = None, theme: Theme = "dark",
    dpi: Dpi = 150,
) -> list[Any]:
    """Bumpy chart (mplsoccer Bumpy): how league positions (or any ranking) change round by round."""
    try:
        png, s = render_bumpy(dataset_id=dataset_id, data=data, filters=filters, x_column=x_column,
                              item_column=item_column, rank_column=rank_column, value_column=value_column,
                              ascending=ascending, x_labels=x_labels, ranks=dict(ranks) if ranks else None,
                              highlight=highlight, theme=theme, title=title, subtitle=subtitle, endnote=endnote,
                              x_label=x_label, y_label=y_label, upside_down=not rank_1_on_top, dpi=dpi)
        return _result(png, s)
    except Exception as exc:  # noqa: BLE001
        raise _err(exc) from exc
