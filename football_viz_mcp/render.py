"""Shared rendering helpers: themes, pitch construction, colours and PNG export."""

from __future__ import annotations

import io
import threading
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from mplsoccer import Pitch, VerticalPitch  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .data import ToolError  # noqa: E402

# matplotlib is not thread safe; all figure work happens under this lock.
RENDER_LOCK = threading.RLock()

PITCH_TYPES = ["statsbomb", "opta", "wyscout", "uefa", "metricasports", "custom", "tracab",
               "skillcorner", "secondspectrum", "impect"]
SIZED_TYPES = {"custom", "tracab", "skillcorner", "secondspectrum", "metricasports"}

THEMES: dict[str, dict[str, Any]] = {
    "light": {"background": "#FFFFFF", "pitch_color": "#FFFFFF", "line_color": "#1F2A2E", "text": "#111111",
              "muted": "#666666", "palette": ["#D7263D", "#1B998B", "#2E86DE", "#F49D37", "#7D4E9E", "#3F4045",
                                              "#E5C100", "#8C564B", "#17BECF", "#BC5090"],
              "cmap": "Reds"},
    "dark": {"background": "#22312B", "pitch_color": "#22312B", "line_color": "#C7D5CC", "text": "#F2F2F2",
             "muted": "#B5C2BA", "palette": ["#FF4F5E", "#43E8D8", "#FFD23F", "#7BA7FF", "#FF9F1C", "#C77DFF",
                                             "#F2F2F2", "#8DE969", "#FF6FB5", "#4CC9F0"],
             "cmap": "hot"},
    "grass": {"background": "#AABB97", "pitch_color": "grass", "line_color": "#FFFFFF", "text": "#111111",
              "muted": "#333333", "palette": ["#D7263D", "#0B3D91", "#FFD23F", "#FFFFFF", "#7D4E9E", "#111111",
                                              "#F49D37", "#17BECF", "#8C564B", "#BC5090"],
              "cmap": "Reds"},
    "night": {"background": "#0E1117", "pitch_color": "#0E1117", "line_color": "#5B6370", "text": "#E6E6E6",
              "muted": "#9AA0A6", "palette": ["#00D1FF", "#FF3C6E", "#FFC857", "#7CFFB2", "#B983FF", "#FF8C42",
                                              "#E6E6E6", "#4D96FF", "#F15BB5", "#9EF01A"],
              "cmap": "magma"},
}

ThemeName = Literal["light", "dark", "grass", "night"]


class PitchConfig(BaseModel):
    """How the pitch looks and which coordinate system the data uses."""

    pitch_type: Literal["statsbomb", "opta", "wyscout", "uefa", "metricasports", "custom", "tracab",
                        "skillcorner", "secondspectrum", "impect"] = Field(
        default="statsbomb",
        description="Coordinate system of the data. statsbomb 120x80 (y down), opta 100x100 (y up), "
                    "wyscout 100x100 (y down), uefa 105x68 m, metricasports 0-1, custom (metres, needs "
                    "pitch_length/pitch_width), tracab (cm, centre origin), skillcorner/secondspectrum/impect "
                    "(m, centre origin).")
    vertical: bool = Field(default=False, description="Draw a vertical pitch (attacking upwards).")
    half: bool = Field(default=False, description="Only draw the attacking half (great for shot maps).")
    pitch_length: float | None = Field(default=None, description="Length in metres for custom/tracab/skillcorner/"
                                                                  "secondspectrum/metricasports (default 105).")
    pitch_width: float | None = Field(default=None, description="Width in metres for sized pitch types (default 68).")
    pitch_color: str | None = Field(default=None, description="Pitch fill colour, or 'grass' for a grass texture.")
    line_color: str | None = Field(default=None, description="Colour of the pitch markings.")
    linewidth: float | None = Field(default=None, description="Width of the pitch markings.")
    stripe: bool = Field(default=False, description="Mowing stripes.")
    stripe_color: str | None = None
    goal_type: Literal["line", "box", "circle"] = "line"
    corner_arcs: bool = False
    spot_type: Literal["circle", "square"] = "circle"
    positional: bool = Field(default=False, description="Draw Juego de Posición (positional play) lines.")
    positional_color: str | None = None
    shade_middle: bool = Field(default=False, description="Shade the central channel (with positional=True).")
    shade_color: str | None = None
    label: bool = Field(default=False, description="Show axis tick labels with coordinates (useful for debugging).")
    tick: bool = False
    axis: bool = False
    pad_left: float | None = None
    pad_right: float | None = None
    pad_top: float | None = None
    pad_bottom: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict,
                                  description="Any other mplsoccer Pitch() keyword argument, e.g. "
                                              "{'line_alpha':0.5,'linestyle':'--','goal_alpha':0.6}.")


def theme_of(name: str | None) -> dict[str, Any]:
    return THEMES.get(name or "light", THEMES["light"])


def make_pitch(cfg: PitchConfig | dict | None, theme: dict[str, Any], line_zorder: float = 2) -> Pitch:
    if cfg is None:
        cfg = PitchConfig()
    elif isinstance(cfg, dict):
        cfg = PitchConfig(**cfg)
    kw: dict[str, Any] = {
        "pitch_type": cfg.pitch_type,
        "half": cfg.half,
        "pitch_color": cfg.pitch_color or theme["pitch_color"],
        "line_color": cfg.line_color or theme["line_color"],
        "stripe": cfg.stripe,
        "goal_type": cfg.goal_type,
        "corner_arcs": cfg.corner_arcs,
        "spot_type": cfg.spot_type,
        "positional": cfg.positional,
        "shade_middle": cfg.shade_middle,
        "label": cfg.label,
        "tick": cfg.tick,
        "axis": cfg.axis,
        "line_zorder": line_zorder,
    }
    if cfg.linewidth is not None:
        kw["linewidth"] = cfg.linewidth
    if cfg.stripe_color:
        kw["stripe_color"] = cfg.stripe_color
    if cfg.positional_color:
        kw["positional_color"] = cfg.positional_color
    if cfg.shade_color:
        kw["shade_color"] = cfg.shade_color
    for side in ("left", "right", "top", "bottom"):
        val = getattr(cfg, f"pad_{side}")
        if val is not None:
            kw[f"pad_{side}"] = val
    if cfg.pitch_type in SIZED_TYPES:
        kw["pitch_length"] = cfg.pitch_length or 105
        kw["pitch_width"] = cfg.pitch_width or 68
    elif cfg.pitch_length or cfg.pitch_width:
        # fixed-size types ignore dimensions in mplsoccer; only pass them where allowed
        pass
    kw.update(cfg.extra or {})
    cls = VerticalPitch if cfg.vertical else Pitch
    try:
        return cls(**kw)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Invalid pitch configuration: {exc}") from exc


def check_colormap(name: str | None, default: str) -> str:
    name = name or default
    if name in colormaps or name in ("grass",):
        return name
    raise ToolError(f"Unknown colormap '{name}'. Examples: Reds, Blues, viridis, magma, hot, coolwarm, RdYlGn, "
                    "YlOrRd, Greens, plasma, inferno, cividis, Spectral (append _r to reverse).")


def check_color(c: str | None) -> str | None:
    if c is None:
        return None
    if mcolors.is_color_like(c):
        return c
    raise ToolError(f"'{c}' is not a valid colour. Use a name ('red') or hex ('#D7263D').")


def is_continuous(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique(dropna=True) > 8


def category_colors(values: list[Any], theme: dict[str, Any], color_map: dict[str, str] | None) -> dict[Any, str]:
    palette = theme["palette"]
    lower_map = {str(k).lower(): v for k, v in (color_map or {}).items()}
    out: dict[Any, str] = {}
    i = 0
    for v in values:
        c = lower_map.get(str(v).lower())
        if c:
            out[v] = check_color(c)  # type: ignore[assignment]
        else:
            out[v] = palette[i % len(palette)]
            i += 1
    return out


def ordered_categories(s: pd.Series, order: list[Any] | None = None) -> list[Any]:
    vals = list(pd.unique(s.dropna()))
    if order:
        lower = {str(v).lower(): v for v in vals}
        first = [lower[str(o).lower()] for o in order if str(o).lower() in lower]
        return first + [v for v in vals if v not in first]
    counts = s.value_counts()
    return [v for v in counts.index if v in vals]


def scale_sizes(values: np.ndarray, size_range: tuple[float, float], vmax: float | None = None) -> np.ndarray:
    v = np.nan_to_num(values.astype(float), nan=0.0)
    lo, hi = size_range
    top = vmax if vmax is not None else (np.nanmax(np.abs(v)) if v.size else 1)
    top = top if top and top > 0 else 1
    return lo + (hi - lo) * np.clip(np.abs(v) / top, 0, 1)


def fig_to_png(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=max(50, min(int(dpi), 300)), facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buf.getvalue()


def add_texts(axs: dict, theme: dict[str, Any], title: str | None, subtitle: str | None, endnote: str | None,
              title_size: float = 26, ha: str = "left") -> None:
    tax = axs.get("title")
    if tax is not None:
        x = 0.0 if ha == "left" else 0.5
        if title:
            tax.text(x, 0.62 if subtitle else 0.45, title, ha=ha, va="center", fontsize=title_size,
                     fontweight="bold", color=theme["text"], transform=tax.transAxes)
        if subtitle:
            tax.text(x, 0.18, subtitle, ha=ha, va="center", fontsize=title_size * 0.6, color=theme["muted"],
                     transform=tax.transAxes)
    eax = axs.get("endnote")
    if eax is not None and endnote:
        eax.text(1.0, 0.5, endnote, ha="right", va="center", fontsize=title_size * 0.5, color=theme["muted"],
                 transform=eax.transAxes)
