"""Reference text served by the get_reference tool (kept out of tool descriptions to save tokens)."""

WORKFLOW = """\
WORKFLOW
1. The user supplies data. Load it with load_dataset (csv_text / json_data / url / file_base64), or the user
   uploads a file at <server>/upload and gives you the dataset_id. Small data can also go inline as `data`
   (a list of records) in any chart tool.
2. Call describe_dataset to learn the column names, value categories and a suggested pitch_type.
3. Pick a chart tool. High-level tools (shot_map, pass_map, heatmap, pass_network, ...) cover common
   requests; pitch_chart composes any layers for anything custom. Every pitch tool accepts
   filters=[{column, op, value}] so you never need to pre-filter the data.
4. Pass the right pitch.pitch_type for the data's coordinates (statsbomb 120x80, opta/wyscout 0-100, ...).
   If points land outside the pitch the tool returns a warning: change pitch_type or the columns.
5. Each chart tool returns a PNG image plus a short text summary with the numbers it plotted — use it to
   describe the chart. If PUBLIC_BASE_URL is configured a download link is included too.
"""

PITCH_TYPES = """\
PITCH TYPES (pitch.pitch_type)               x range        y range           notes
statsbomb                                    0-120          0-80              y=0 at the TOP
opta                                         0-100          0-100             y=0 at the bottom
wyscout                                      0-100          0-100             y=0 at the top
uefa                                         0-105          0-68              metres
metricasports                                0-1            0-1               y=0 at the top; set pitch_length/width
custom                                       0-length       0-width           metres; set pitch_length/pitch_width
tracab                                       ±length*50     ±width*50         centimetres, origin at centre
skillcorner / secondspectrum / impect        ±length/2      ±width/2          metres, origin at centre
Use transform_dataset(standardize_from, standardize_to) to convert coordinates between systems.
Other pitch options: vertical, half, pitch_color ('grass' texture), line_color, stripe, goal_type
(line|box|circle), corner_arcs, positional (Juego de Posición lines), shade_middle, label/tick (show axes),
pad_left/right/top/bottom, extra={any mplsoccer Pitch kwarg}.
Themes (figure.theme): light, dark, grass, night.
"""

LAYERS = """\
LAYERS for pitch_chart (common fields: type, dataset_id|data, filters, x, y, end_x, end_y, color_by, size_by,
label_by, group_by, values, color, color_map, cmap, alpha, size, marker, label, mirror, zorder, options, style).
Coordinates columns are auto-detected (x/y, location_x/location_y, end_x/end_y, pass.end_location_x, ...).
color_by: categorical -> one colour per value + legend (override with color_map); numeric -> cmap + colorbar.
style: raw matplotlib kwargs for the underlying call (e.g. {"edgecolors":"black","linewidths":2}).

scatter        points. size_by scales size (options.size_range [min,max], size_max). marker: o s ^ * h X
               football boot_left boot_right shirt arrowhead. options.rotation_by (degrees column),
               options.edgecolor, label_by + options.label_size/label_color/label_offset.
arrows         start->end arrows. size = width. options.headwidth, headlength, show_start.
lines          start->end lines. options.comet (tapered), transparent, alpha_start, alpha_end, show_end.
comet          shortcut for lines with comet=true + transparent fade + end dots.
plot           connect points in order (paths/trajectories). group_by = one path per group,
               options.sort_by (e.g. timestamp), marker.
kde            smooth density. cmap, options.fill (true), levels (100), thresh (0.05), cut (4), bw_adjust (1).
hexbin         hexagon density. options.gridsize [nx, ny], mincnt (1), edgecolor; values + options.statistic.
heatmap        grid heatmap via bin_statistic. options.bins [nx, ny] (6,5), statistic count|mean|sum|median|
               std|min|max (needs values), normalize (share of total), smooth (gaussian sigma), vmin, vmax,
               show_labels, label_format ('{:.0%}', '{:.0f}', '{:.2f}'), label_size, label_color, exclude_zeros.
heatmap_positional  Juego de Posición zones. options.positional full|horizontal|vertical + heatmap options.
heatmap_zones  custom zones. options.zones = 'positional_full'|'positional_horizontal'|'positional_vertical'
               or [[x0,x1,y0,y1], ...] tiling the pitch exactly; options.names; show_labels.
zones          just draw zones (no data). options.zones, names, facecolor, show_names.
flow           flow map of average direction per bin. options.bins [nx,ny], arrow_type average|same|scale,
               arrow_length, headwidth; color or cmap.
sonar          pass sonars per grid cell (or zone). Needs end_x/end_y, or options.angle_column
               (+ angle_degrees). options.bins [nx,ny], angle_bins (10), zones (as heatmap_zones),
               color_by_stat distance|values|none, cmap, rmin, center.
pass_network   group_by = passer column, options.recipient = receiver column; x,y (+end_x,end_y improves
               positions). options.min_passes (3), players [subset], undirected (true), label
               name|last_name|initials|none, labels {player: 'short'}, label_inside, node_color, edge_color,
               node_size_range, max_edge_width, edge_alpha_by_count.
voronoi        freeze-frame control areas. group_by = team column, options.team_a (value of first team),
               show_players (true), color_map per team.
delaunay       Delaunay triangulation lines (group_by for one per team).
convexhull     team/player shape polygons. group_by (one hull per group), options.trim_outliers (e.g. 0.9
               keeps 90% closest to median), show_points.
goal_angle     shaded shooting angle to goal for each row. options.goal right|left, max_shots.
text           write label_by text at x,y.
annotate       (no data) options.annotations=[{text,x,y,text_x?,text_y?,arrow?,curve?,color?,size?,bold?,box?}].
polygon        (no data) options.polygons=[[[x,y],...],...] or options.vertices.
formation      (no data) options.formation e.g. '433','4231','352' (see list_formations),
               options.players {position: name} or [names in position order], numbers {position: n},
               flip (attack right->left), half, label_size; color, size, marker ('shirt' looks nice).

FIGURE options: title, subtitle, endnote, theme, background, figheight (inches), dpi, legend, legend_loc
('upper left', 'lower right', 'below'...), colorbar, colorbar_label, facet_by (one pitch per value of a column,
e.g. player or half), facet_order, ncols, max_facets, font_scale.
"""

OTHER = """\
NON-PITCH CHARTS
radar_chart   params + values (+ compare_values) + min_range/max_range, or dataset_id + name_column + player
              (+ compare_player) + params (stat columns): ranges auto = 5th-95th percentile of the dataset
              (narrow the comparison pool with filters). style standard|solid|turbine (turbine needs a dataset).
              lower_is_better [params], param_labels (display names), num_rings, curved_labels.
pizza_chart   percentile pizza. params + values (0-100) or dataset mode with as_percentiles=true
              (percentile rank inside the filtered pool, show_raw_values prints the raw stat).
              param_groups {group: [params]} + group_colors colours slices by category (attack/defence...).
              compare_player / compare_values overlays a second player.
bumpy_chart   rank evolution. dataset (x_column round/week, item_column team, rank_column or value_column)
              or x_labels + ranks {team: [rank per round]}. highlight {team: colour}. upside_down true = 1 at top.
joint_plot    pitch with marginal distributions (kind kde|scatter|hexbin, marginal hist|kde).
zone_stats    numbers only (no image): counts / means per grid cell or positional zone, for text answers.
"""

FORMATIONS_NOTE = "Formations accept with or without dashes, e.g. '4-3-3' or '433'."


def reference(topic: str) -> str:
    topics = {"workflow": WORKFLOW, "pitch": PITCH_TYPES, "layers": LAYERS, "other": OTHER}
    if topic == "all":
        return "\n".join(topics.values())
    return topics.get(topic, "\n".join(topics.values()))
