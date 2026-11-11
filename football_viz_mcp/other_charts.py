"""Non-pitch mplsoccer charts: radar, pizza, bumpy chart, and pitch joint plots."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from mplsoccer import Bumpy, PyPizza, Radar, grid

from .data import Filter, ToolError, apply_filters, auto_column, get_df, numeric, resolve_column
from .render import (RENDER_LOCK, PitchConfig, add_texts, check_color, check_colormap, fig_to_png, make_pitch,
                     theme_of)


def _pick_row(df: pd.DataFrame, name_col: str, who: str) -> pd.Series:
    s = df[name_col].astype(str)
    exact = df[s.str.lower() == who.lower()]
    if len(exact) == 0:
        exact = df[s.str.lower().str.contains(who.lower(), regex=False)]
    if len(exact) == 0:
        raise ToolError(f"'{who}' not found in column '{name_col}'. Examples: {list(s.unique()[:10])}")
    if len(exact) > 1:
        names = list(exact[name_col].astype(str).unique())
        if len(names) > 1:
            raise ToolError(f"'{who}' matches several rows: {names[:10]}. Be more specific.")
    return exact.iloc[0]


def _params_from_dataset(dataset_id, data, filters, name_column, player, compare_player, params):
    df = get_df(dataset_id, data, None)
    pool = apply_filters(df, filters)  # comparison pool (e.g. same position / minutes)
    name_col = resolve_column(df, name_column, "name_column")
    cols = [resolve_column(df, p, "param") for p in params]
    row = _pick_row(df, name_col, player)
    comp = _pick_row(df, name_col, compare_player) if compare_player else None
    return df, pool, name_col, cols, row, comp


# --------------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------------- #


def render_radar(*, params: list[str] | None, values: list[float] | None, compare_values: list[float] | None,
                 min_range: list[float] | None, max_range: list[float] | None, lower_is_better: list[str] | None,
                 dataset_id: str | None, data: list[dict] | None, filters: list[Filter] | None,
                 name_column: str | None, player: str | None, compare_player: str | None,
                 player_label: str | None, compare_label: str | None,
                 style: Literal["standard", "solid", "turbine"], theme: str, color: str | None,
                 compare_color: str | None, title: str | None, subtitle: str | None, endnote: str | None,
                 param_labels: list[str] | None, num_rings: int, round_int: bool | None, curved_labels: bool,
                 dpi: int) -> tuple[bytes, str]:
    th = theme_of(theme)
    dist = None
    if dataset_id or data:
        if not (name_column and player and params):
            raise ToolError("With a dataset, give name_column, player and params (the stat columns).")
        df, pool, name_col, cols, row, comp = _params_from_dataset(dataset_id, data, filters, name_column, player,
                                                                   compare_player, params)
        values = [float(row[c]) for c in cols]
        compare_values = [float(comp[c]) for c in cols] if comp is not None else compare_values
        pool_vals = pool[cols].apply(pd.to_numeric, errors="coerce")
        if min_range is None:
            min_range = [float(v) for v in pool_vals.quantile(0.05)]
        if max_range is None:
            max_range = [float(v) for v in pool_vals.quantile(0.95)]
        dist = pool_vals.dropna().to_numpy()
        player_label = player_label or str(row[name_col])
        if comp is not None:
            compare_label = compare_label or str(comp[name_col])
        params_labels = param_labels or params
    else:
        if not params or values is None:
            raise ToolError("Give params and values (or a dataset_id with name_column/player).")
        params_labels = param_labels or params
        if min_range is None or max_range is None:
            allv = np.array([values] + ([compare_values] if compare_values else []), dtype=float)
            min_range = min_range or [0.0] * len(params)
            max_range = max_range or [float(v) * 1.15 if v > 0 else 1.0 for v in allv.max(axis=0)]
    n = len(params_labels)
    if not (len(values) == len(min_range) == len(max_range) == n):
        raise ToolError("params, values, min_range and max_range must have the same length.")
    if compare_values is not None and len(compare_values) != n:
        raise ToolError("compare_values must have the same length as params.")
    for i in range(n):  # mplsoccer requires min != max
        if min_range[i] == max_range[i]:
            max_range[i] = min_range[i] + 1
    lib = [params_labels[params.index(p)] if p in params else p for p in (lower_is_better or [])] \
        if params else lower_is_better
    if round_int is None:
        round_int = [bool(float(a).is_integer() and float(b).is_integer()) for a, b in zip(min_range, max_range)]
    else:
        round_int = [round_int] * n
    radar = Radar(params_labels, min_range, max_range, lower_is_better=lib, round_int=round_int,
                  num_rings=num_rings, ring_width=1, center_circle_radius=1)
    c1 = check_color(color) or th["palette"][0]
    c2 = check_color(compare_color) or th["palette"][1]
    ring_fc = "#E6E6E6" if theme in ("light", "grass") else "#34433C" if theme == "dark" else "#1D232D"
    with RENDER_LOCK:
        fig, axs = grid(figheight=12, grid_height=0.86, title_height=0.08, endnote_height=0.03, title_space=0,
                        endnote_space=0, grid_key="radar", axis=False)
        fig.set_facecolor(th["background"])
        ax = axs["radar"]
        radar.setup_axis(ax=ax, facecolor="None")
        radar.draw_circles(ax=ax, facecolor=ring_fc, edgecolor=th["background"], lw=1.5)
        if style == "solid":
            radar_out = radar.draw_radar_solid(values, ax=ax, kwargs={"facecolor": c1, "alpha": 0.6,
                                                                       "edgecolor": c1, "lw": 3})
            if compare_values is not None:
                radar.draw_radar_solid(compare_values, ax=ax, kwargs={"facecolor": c2, "alpha": 0.6,
                                                                      "edgecolor": c2, "lw": 3})
        elif style == "turbine":
            pass
        elif compare_values is not None:
            radar.draw_radar_compare(values, compare_values, ax=ax, kwargs_radar={"facecolor": c1, "alpha": 0.6},
                                     kwargs_compare={"facecolor": c2, "alpha": 0.6})
        else:
            radar.draw_radar(values, ax=ax, kwargs_radar={"facecolor": c1, "alpha": 0.75},
                             kwargs_rings={"facecolor": c2, "alpha": 0.6})
        if style == "turbine":
            if dist is None or len(dist) < 5:
                raise ToolError("style='turbine' needs a dataset with at least 5 rows (the distribution).")
            radar.turbine(values, dist, ax=ax, kwargs_inner={"edgecolor": th["muted"], "color": c1},
                          kwargs_outer={"edgecolor": th["muted"], "color": ring_fc})
        radar.draw_range_labels(ax=ax, fontsize=15, color=th["text"])
        radar.draw_param_labels(ax=ax, fontsize=17, color=th["text"], curved=curved_labels)
        tax = axs["title"]
        if player_label:
            tax.text(0.01, 0.65, player_label, fontsize=26, color=c1, fontweight="bold", ha="left", va="center")
        if compare_values is not None and compare_label:
            tax.text(0.99, 0.65, compare_label, fontsize=26, color=c2, fontweight="bold", ha="right", va="center")
        if title:
            tax.text(0.5 if compare_values is not None else 0.01, 0.2 if player_label else 0.5, title,
                     fontsize=18, color=th["muted"],
                     ha="center" if compare_values is not None else "left", va="center")
        if subtitle:
            tax.text(0.99 if compare_values is None else 0.5, 0.2 if compare_values is None else -0.1, subtitle,
                     fontsize=15, color=th["muted"], ha="right" if compare_values is None else "center", va="center")
        add_texts({"endnote": axs["endnote"]}, th, None, None, endnote)
        png = fig_to_png(fig, dpi)
    summ = ", ".join(f"{p}={v:g}" for p, v in zip(params_labels, values))
    if compare_values is not None:
        summ += " | compare: " + ", ".join(f"{p}={v:g}" for p, v in zip(params_labels, compare_values))
    return png, f"radar ({style}): {summ}"


# --------------------------------------------------------------------------- #
# Pizza
# --------------------------------------------------------------------------- #


def render_pizza(*, params: list[str] | None, values: list[float] | None, compare_values: list[float] | None,
                 min_range: list[float] | None, max_range: list[float] | None, dataset_id: str | None,
                 data: list[dict] | None, filters: list[Filter] | None, name_column: str | None,
                 player: str | None, compare_player: str | None, as_percentiles: bool,
                 lower_is_better: list[str] | None, show_raw_values: bool, param_labels: list[str] | None,
                 param_groups: dict[str, list[str]] | None, group_colors: dict[str, str] | None,
                 slice_color: str | None, compare_color: str | None, theme: str, title: str | None,
                 subtitle: str | None, endnote: str | None, curved_labels: bool, dpi: int) -> tuple[bytes, str]:
    th = theme_of(theme)
    alt_values = None
    alt_compare = None
    if dataset_id or data:
        if not (name_column and player and params):
            raise ToolError("With a dataset, give name_column, player and params (the stat columns).")
        df, pool, name_col, cols, row, comp = _params_from_dataset(dataset_id, data, filters, name_column, player,
                                                                   compare_player, params)
        raw = [float(row[c]) for c in cols]
        raw_c = [float(comp[c]) for c in cols] if comp is not None else None
        if as_percentiles:
            pool_vals = pool[cols].apply(pd.to_numeric, errors="coerce")
            lib = {resolve_column(df, p, "lower_is_better") for p in (lower_is_better or [])}

            def pct(v, c):
                arr = pool_vals[c].dropna().to_numpy()
                if arr.size == 0:
                    return 0
                p = (arr < v).mean() * 100 + (arr == v).mean() * 50
                return int(round(100 - p if c in lib else p))

            values = [pct(v, c) for v, c in zip(raw, cols)]
            compare_values = [pct(v, c) for v, c in zip(raw_c, cols)] if raw_c else None
            min_range, max_range = [0] * len(cols), [100] * len(cols)
            if show_raw_values:
                alt_values = [f"{v:g}" for v in raw]
                alt_compare = [f"{v:g}" for v in raw_c] if raw_c else None
        else:
            values, compare_values = raw, raw_c
            pool_vals = pool[cols].apply(pd.to_numeric, errors="coerce")
            min_range = min_range or [float(v) for v in pool_vals.min()]
            max_range = max_range or [float(v) for v in pool_vals.max()]
        title = title or str(row[name_col]) + (f" vs {comp[name_col]}" if comp is not None else "")
    else:
        if not params or values is None:
            raise ToolError("Give params and values (percentiles 0-100 by default) or a dataset_id.")
    labels = param_labels or params
    n = len(labels)
    if len(values) != n:
        raise ToolError("params and values must have the same length.")
    min_range = min_range or [0] * n
    max_range = max_range or [100] * n
    values = [int(round(v)) if float(v).is_integer() or (min_range == [0] * n and max_range == [100] * n) else v
              for v in values]
    # colours per slice
    base = check_color(slice_color) or th["palette"][0]
    slice_colors = [base] * n
    legend_groups: list[tuple[str, str]] = []
    if param_groups:
        gcols = group_colors or {}
        pal = th["palette"]
        for gi, (g, plist) in enumerate(param_groups.items()):
            c = check_color(gcols.get(g)) or pal[gi % len(pal)]
            legend_groups.append((g, c))
            low = {str(p).lower() for p in plist}
            for i, p in enumerate(params):
                if str(p).lower() in low or str(labels[i]).lower() in low:
                    slice_colors[i] = c
    text_color = th["text"]
    bg = th["background"]
    line_c = "#CCCCCC" if theme in ("light", "grass") else "#444C48"
    with RENDER_LOCK:
        baker = PyPizza(params=labels, min_range=min_range, max_range=max_range, background_color=bg,
                        straight_line_color=line_c, straight_line_lw=1, last_circle_color=line_c, last_circle_lw=1,
                        other_circle_lw=0, other_circle_color=line_c, inner_circle_size=20)
        kw: dict[str, Any] = dict(
            figsize=(9, 10), param_location=110, color_blank_space="same", blank_alpha=0.35,
            kwargs_slices=dict(edgecolor=bg, zorder=2, linewidth=1),
            kwargs_params=dict(color=text_color, fontsize=12, va="center"),
            kwargs_values=dict(color="#000000" if theme in ("light", "grass") else "#FFFFFF", fontsize=11, zorder=3,
                               bbox=dict(edgecolor="#000000", facecolor=base, boxstyle="round,pad=0.2", lw=1)),
            curved_params=curved_labels)
        if compare_values is not None:
            c2 = check_color(compare_color) or th["palette"][1]
            kw.pop("color_blank_space")
            kw["kwargs_compare"] = dict(facecolor=c2, edgecolor=bg, zorder=2, linewidth=1)
            kw["kwargs_compare_values"] = dict(color="#000000", fontsize=11, zorder=3,
                                               bbox=dict(edgecolor="#000000", facecolor=c2,
                                                         boxstyle="round,pad=0.2", lw=1))
            kw["kwargs_slices"]["facecolor"] = base
            fig, ax = baker.make_pizza(values, compare_values=compare_values, alt_text_values=alt_values,
                                       alt_text_compare_values=alt_compare, **kw)
            params_offset = [False] * n
            baker.adjust_texts(params_offset, offset=-0.2, adj_comp_values=True)
        else:
            fig, ax = baker.make_pizza(values, alt_text_values=alt_values, slice_colors=slice_colors,
                                       value_bck_colors=slice_colors, **kw)
        fig.set_facecolor(bg)
        if title:
            fig.text(0.515, 0.985, title, size=20, ha="center", color=text_color, fontweight="bold")
        if subtitle:
            fig.text(0.515, 0.955, subtitle, size=13, ha="center", color=th["muted"])
        if endnote:
            fig.text(0.99, 0.01, endnote, size=10, color=th["muted"], ha="right")
        if compare_values is not None:
            names = [str(row[name_col]), str(comp[name_col])] if (dataset_id or data) and comp is not None \
                else ["", ""]
            if names[0]:
                fig.text(0.35, 0.925, names[0], size=13, ha="center", color="#FFFFFF",
                         bbox=dict(facecolor=base, edgecolor="none", boxstyle="round,pad=0.3"))
                fig.text(0.68, 0.925, names[1], size=13, ha="center", color="#FFFFFF",
                         bbox=dict(facecolor=c2, edgecolor="none", boxstyle="round,pad=0.3"))
        if legend_groups and compare_values is None:
            x = 0.5 - 0.12 * (len(legend_groups) - 1) / 2
            for i, (g, c) in enumerate(legend_groups):
                fig.text(x + 0.12 * i, 0.925, g, size=12, ha="center", color=text_color,
                         bbox=dict(facecolor=c, edgecolor="none", boxstyle="round,pad=0.3"))
        png = fig_to_png(fig, dpi)
    summ = ", ".join(f"{p}={v}" for p, v in zip(labels, values))
    if compare_values is not None:
        summ += " | compare: " + ", ".join(f"{p}={v}" for p, v in zip(labels, compare_values))
    return png, f"pizza: {summ}"


# --------------------------------------------------------------------------- #
# Bumpy
# --------------------------------------------------------------------------- #


def render_bumpy(*, dataset_id, data, filters, x_column, item_column, rank_column, value_column,
                 ascending, x_labels, ranks, highlight, theme, title, subtitle, endnote, x_label, y_label,
                 upside_down, dpi) -> tuple[bytes, str]:
    th = theme_of(theme)
    if dataset_id or data:
        df = get_df(dataset_id, data, filters)
        xc = resolve_column(df, x_column, "x_column")
        ic = resolve_column(df, item_column, "item_column")
        if rank_column:
            rc = resolve_column(df, rank_column, "rank_column")
            df = df.assign(_rank=pd.to_numeric(df[rc], errors="coerce"))
        elif value_column:
            vc = resolve_column(df, value_column, "value_column")
            df = df.assign(_v=pd.to_numeric(df[vc], errors="coerce"))
            df["_rank"] = df.groupby(xc)["_v"].rank(ascending=ascending, method="first")
        else:
            raise ToolError("Give rank_column (positions) or value_column (to rank by).")
        order = list(pd.unique(df[xc]))
        try:
            order = sorted(order)
        except TypeError:
            pass
        piv = df.pivot_table(index=ic, columns=xc, values="_rank", aggfunc="first").reindex(columns=order)
        x_labels = [str(v) for v in order]
        ranks = {str(k): [int(v) if pd.notna(v) else None for v in row] for k, row in piv.iterrows()}
    if not x_labels or not ranks:
        raise ToolError("Give x_labels + ranks ({item: [rank per x]}) or a dataset.")
    n_items = max(int(np.nanmax([v for r in ranks.values() for v in r if v is not None])), len(ranks))
    for k, r in ranks.items():
        if len(r) != len(x_labels):
            raise ToolError(f"ranks['{k}'] has {len(r)} values but there are {len(x_labels)} x_labels.")
        if any(v is None for v in r):
            # fill gaps by carrying the previous rank
            last = next((v for v in r if v is not None), n_items)
            ranks[k] = [last := (v if v is not None else last) for v in r]  # noqa: F841
    hl = {}
    pal = th["palette"]
    for i, (k, c) in enumerate((highlight or {}).items()):
        match = next((r for r in ranks if r.lower() == str(k).lower()), None)
        if match is None:
            raise ToolError(f"highlight item '{k}' not in data. Items: {list(ranks)[:30]}")
        hl[match] = check_color(c) if c else pal[i % len(pal)]
    if not hl:
        first = sorted(ranks, key=lambda k: ranks[k][-1])[:3]
        hl = {k: pal[i] for i, k in enumerate(first)}
    light = theme in ("light", "grass")
    with RENDER_LOCK:
        bumpy = Bumpy(background_color=th["background"], scatter_color="#BBBBBB" if light else "#4F535C",
                      label_color=th["text"], line_color="#CCCCCC" if light else "#3E4246",
                      rotate_xticks=90 if len(x_labels) > 12 else 0, ticklabel_size=13, label_size=18,
                      scatter_primary="D", show_right=False, plot_labels=True, alignment_yvalue=0.1,
                      alignment_xvalue=0.065)
        y_list = np.linspace(1, n_items, n_items).astype(int)
        fig, ax = bumpy.plot(x_list=x_labels, y_list=y_list, values=ranks, secondary_alpha=0.5,
                             highlight_dict=hl, figsize=(16, max(8, n_items * 0.5)), x_label=x_label or "",
                             y_label=y_label or "Position", ylim=(-0.1, n_items + 1), lw=2.5,
                             upside_down=upside_down)
        xs_end = len(x_labels) - 1
        add = (0 if n_items % 2 == 0 else 1) if upside_down else (1 if n_items % 2 == 0 else 0)
        for k, r in ranks.items():
            yv = (r[-1] + add) if upside_down else (n_items - r[-1] + add)
            col = hl.get(k, th["muted"])
            ax.text(xs_end + 0.35, yv, k, va="center", ha="left", fontsize=12, color=col,
                    fontweight="bold" if k in hl else "normal")
        ax.set_xlim(-0.5, xs_end + 2.2)
        if title:
            fig.text(0.09, 0.99, title, size=24, color=th["text"], fontweight="bold", ha="left")
        if subtitle:
            fig.text(0.09, 0.955, subtitle, size=15, color=th["muted"], ha="left")
        if endnote:
            fig.text(0.99, 0.005, endnote, size=11, color=th["muted"], ha="right")
        png = fig_to_png(fig, dpi)
    final = sorted(ranks.items(), key=lambda kv: kv[1][-1])[:5]
    return png, "bumpy: final top " + ", ".join(f"{k} ({v[-1]})" for k, v in final)


# --------------------------------------------------------------------------- #
# Joint plot
# --------------------------------------------------------------------------- #


def render_joint(*, dataset_id, data, filters, x, y, pitch: PitchConfig | None, kind: str, marginal: str,
                 color: str | None, cmap: str | None, theme: str, title, subtitle, endnote, dpi) -> tuple[bytes, str]:
    th = theme_of(theme)
    df = get_df(dataset_id, data, filters)
    xc, yc = auto_column(df, x, "x"), auto_column(df, y, "y")
    xs, ys = numeric(df, xc), numeric(df, yc)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 3:
        raise ToolError("joint_plot needs at least 3 points.")
    color = check_color(color) or th["palette"][0]
    vertical = bool(pitch and pitch.vertical)
    with RENDER_LOCK:
        p = make_pitch(pitch, th)
        fig, axs = p.jointgrid(figheight=10, grid_height=0.72, title_height=0.1, endnote_height=0.04,
                               marginal=0.12, ax_left=False, ax_top=True, ax_right=True, ax_bottom=False,
                               axis=False, space=0.01)
        fig.set_facecolor(th["background"])
        ax = axs["pitch"]
        if kind == "kde":
            p.kdeplot(xs, ys, ax=ax, fill=True, levels=100, thresh=0.05, cut=4,
                      cmap=check_colormap(cmap, th["cmap"]), zorder=1)
        elif kind == "hexbin":
            p.hexbin(xs, ys, ax=ax, gridsize=(10, 8), cmap=check_colormap(cmap, th["cmap"]),
                     edgecolors=th["background"], mincnt=1)
        else:
            p.scatter(xs, ys, ax=ax, s=80, color=color, edgecolors=th["background"], alpha=0.8, zorder=3)
        # marginals: along-pitch values on the long axis
        long_vals, short_vals = (xs, ys)
        top_vals, right_vals = (short_vals, long_vals) if vertical else (long_vals, short_vals)
        for key, vals, orient in (("top", top_vals, "vertical"), ("right", right_vals, "horizontal")):
            max_ = axs[key]
            if marginal == "kde":
                from scipy.stats import gaussian_kde
                lim = max_.get_xlim() if orient == "vertical" else max_.get_ylim()
                grid_v = np.linspace(min(lim), max(lim), 200)
                dens = gaussian_kde(vals)(grid_v)
                if orient == "vertical":
                    max_.fill_between(grid_v, 0, dens, color=color, alpha=0.6)
                    max_.set_ylim(0, dens.max() * 1.05)
                else:
                    max_.fill_betweenx(grid_v, 0, dens, color=color, alpha=0.6)
                    max_.set_xlim(0, dens.max() * 1.05)
            else:
                lim = max_.get_xlim() if orient == "vertical" else max_.get_ylim()
                bins = np.linspace(min(lim), max(lim), 25)
                max_.hist(vals, bins=bins, color=color, alpha=0.8, orientation=orient,
                          edgecolor=th["background"])
            max_.set_facecolor(th["background"])
        add_texts(axs, th, title, subtitle, endnote)
        png = fig_to_png(fig, dpi)
    return png, f"joint plot ({kind} + {marginal} marginals): {xs.size} points"
