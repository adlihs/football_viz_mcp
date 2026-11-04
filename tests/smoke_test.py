"""End-to-end test: talks to a running server over Streamable HTTP and calls every tool.

Usage: python tests/smoke_test.py http://localhost:8000/mcp?key=KEY [output_dir]
"""
import asyncio
import base64
import json
import sys
from pathlib import Path

from mcp import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/mcp"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "test_output")
OUT.mkdir(parents=True, exist_ok=True)
EX = Path(__file__).resolve().parent.parent / "examples"
FAIL = []


def save(name, res):
    imgs = [c for c in res.content if getattr(c, "type", "") == "image"]
    txt = " ".join(c.text for c in res.content if getattr(c, "type", "") == "text")
    if res.is_error:
        FAIL.append(name)
        print(f"✗ {name}: {txt[:400]}")
        return txt
    for i, c in enumerate(imgs):
        (OUT / f"{name}{'' if i == 0 else i}.png").write_bytes(base64.b64decode(c.data))
    print(f"✓ {name}: {txt[:220].replace(chr(10), ' | ')}")
    return txt


async def main():
    async with Client(URL) as c:
        tools = (await c.list_tools()).tools
        print(len(tools), "tools:", [t.name for t in tools])

        async def call(name, args, label=None):
            res = await c.call_tool(name, args)
            return res, save(label or name, res)

        res, _ = await call("load_dataset", {"name": "partido", "csv_text": (EX / "partido_eventos.csv").read_text()})
        ev = json.loads(res.content[0].text)["dataset_id"] if not res.is_error else None
        print("   suggested:", json.loads(res.content[0].text).get("suggested_pitch_type"))
        b64 = base64.b64encode((EX / "jugadores_temporada.csv").read_bytes()).decode()
        res, _ = await call("load_dataset", {"name": "jugadores", "file_base64": b64, "file_name": "j.csv"})
        pl = json.loads(res.content[0].text)["dataset_id"]
        res, _ = await call("load_dataset", {"name": "tabla", "csv_text": (EX / "tabla_por_jornada.csv").read_text()})
        tb = json.loads(res.content[0].text)["dataset_id"]
        res, _ = await call("load_dataset", {"name": "ff", "csv_text": (EX / "freeze_frame_tiro.csv").read_text()})
        ff = json.loads(res.content[0].text)["dataset_id"]
        await call("list_datasets", {})
        await call("describe_dataset", {"dataset_id": ev, "columns": ["type", "outcome", "x"]})
        await call("get_reference", {"topic": "pitch"})
        await call("list_formations", {})

        sap = [{"column": "team", "value": "Saprissa"}]
        passes = sap + [{"column": "type", "value": "Pass"}]
        shots = [{"column": "type", "value": "Shot"}]

        await call("draw_pitch", {"pitch": {"pitch_type": "uefa", "pitch_color": "grass", "stripe": True,
                                            "line_color": "white"}, "title": "Cancha", "theme": "grass"})
        await call("draw_pitch", {"pitch": {"positional": True, "shade_middle": True, "vertical": True}},
                   "draw_pitch_positional")
        await call("shot_map", {"dataset_id": ev, "filters": shots, "xg": "xg", "outcome": "outcome",
                                "title": "Remates", "subtitle": "Saprissa vs Alajuelense", "endnote": "@coach"})
        await call("shot_map", {"dataset_id": ev, "filters": shots, "xg": "xg", "outcome": "outcome",
                                "facet_by": "team", "theme": "dark", "show_kde": True}, "shot_map_facets")
        await call("pass_map", {"dataset_id": ev, "filters": passes + [{"column": "player", "value": "Bennette"}],
                                "color_by": "outcome", "color_map": {"Complete": "#1B998B", "Incomplete": "#D7263D"},
                                "title": "Pases de Bennette"})
        await call("pass_map", {"dataset_id": ev, "filters": passes + [{"column": "end_x", "op": ">", "value": 80}],
                                "style": "comet", "theme": "night", "background": "kde"}, "pass_map_comet")
        await call("pass_map", {"dataset_id": ev, "filters": passes, "facet_by": "player", "style": "lines",
                                "color_by": "outcome", "legend_loc": "below"}, "pass_map_facets")
        for kind in ["grid", "positional", "zones", "kde", "hexbin"]:
            await call("heatmap", {"dataset_id": ev, "filters": sap, "kind": kind, "show_labels": True,
                                   "normalize": kind in ("grid", "positional"), "title": f"heatmap {kind}"},
                       f"heatmap_{kind}")
        await call("heatmap", {"dataset_id": ev, "filters": sap, "bins": [18, 12], "smooth": 1.2,
                               "pitch": {"vertical": True}, "theme": "dark", "cmap": "hot"}, "heatmap_smooth")
        await call("flow_map", {"dataset_id": ev, "filters": passes, "title": "Flow"})
        await call("pass_network", {"dataset_id": ev, "filters": [{"column": "type", "value": "Pass"},
                                                                  {"column": "outcome", "value": "Complete"}],
                                    "passer": "player", "recipient": "recipient", "facet_by": "team",
                                    "min_passes": 4, "title": "Redes de pase"})
        await call("pass_sonar", {"dataset_id": ev, "filters": passes, "title": "Sonar"})
        await call("pass_sonar", {"dataset_id": ev, "filters": passes, "positional_zones": True}, "pass_sonar_zones")
        await call("freeze_frame", {"dataset_id": ff, "team": "equipo", "team_a": "Saprissa", "label_by": "jugador",
                                    "ball": [108, 44], "shot_angle": True, "show_hull": True,
                                    "color_map": {"Saprissa": "#7D4E9E", "Alajuelense": "#D7263D"},
                                    "pitch": {"half": True}})
        await call("formation_chart", {"formation": "4-3-3", "players": {"GK": "Navas", "ST": "Campbell"},
                                       "numbers": {"GK": 1, "ST": 9}, "marker": "shirt", "title": "Alineación"})
        await call("formation_chart", {"formation": "433", "opponent_formation": "4231",
                                       "team_name": "Saprissa", "opponent_name": "Alajuelense"}, "formation_vs")
        await call("joint_plot", {"dataset_id": ev, "filters": sap, "kind": "kde", "marginal": "kde"})
        await call("zone_stats", {"dataset_id": ev, "filters": sap, "group_by": "half"})
        await call("radar_chart", {"dataset_id": pl, "name_column": "jugador", "player": "Campbell",
                                   "compare_player": "Ugalde",
                                   "params": ["goles_90", "xg_90", "asist_90", "pases_clave_90", "regates_90",
                                              "perdidas_90"], "lower_is_better": ["perdidas_90"],
                                   "filters": [{"column": "posicion", "value": "DEL"}], "title": "Delanteros"})
        await call("radar_chart", {"params": ["A", "B", "C", "D", "E"], "values": [3, 5, 2, 7, 4],
                                   "min_range": [0] * 5, "max_range": [10] * 5, "style": "solid",
                                   "theme": "dark"}, "radar_inline")
        await call("radar_chart", {"dataset_id": pl, "name_column": "jugador", "player": "Campbell",
                                   "params": ["goles_90", "xg_90", "asist_90", "pases_clave_90", "regates_90"],
                                   "style": "turbine"}, "radar_turbine")
        await call("pizza_chart", {"dataset_id": pl, "name_column": "jugador", "player": "Bennette",
                                   "params": ["goles_90", "asist_90", "pases_clave_90", "pases_prog_90",
                                              "regates_90", "recuperaciones_90", "duelos_aereos_pct",
                                              "precision_pase_pct"],
                                   "param_groups": {"Ataque": ["goles_90", "asist_90", "pases_clave_90"],
                                                    "Posesión": ["pases_prog_90", "regates_90", "precision_pase_pct"],
                                                    "Defensa": ["recuperaciones_90", "duelos_aereos_pct"]},
                                   "show_raw_values": True, "subtitle": "Percentiles vs. liga"})
        await call("pizza_chart", {"dataset_id": pl, "name_column": "jugador", "player": "Bennette",
                                   "compare_player": "Zamora", "params": ["goles_90", "asist_90", "regates_90",
                                                                          "pases_prog_90", "recuperaciones_90"]},
                   "pizza_compare")
        await call("bumpy_chart", {"dataset_id": tb, "x_column": "jornada", "item_column": "equipo",
                                   "rank_column": "posicion", "highlight": {"Saprissa": "#7D4E9E",
                                                                            "Alajuelense": "#D7263D"},
                                   "title": "Tabla por jornada"})
        await call("pitch_chart", {
            "dataset_id": ev,
            "pitch": {"pitch_type": "statsbomb", "pitch_color": "#22312b", "line_color": "#c7d5cc"},
            "figure": {"title": "Composición libre", "theme": "dark", "facet_by": "half"},
            "layers": [
                {"type": "heatmap_positional", "filters": sap, "alpha": 0.7, "cmap": "Greens"},
                {"type": "convexhull", "filters": sap + [{"column": "type", "value": "Pass"}],
                 "group_by": "player", "options": {"order": ["Bennette", "Campbell"], "trim_outliers": 0.8,
                                                   "show_points": False},
                 "filters": sap + [{"column": "player", "op": "in", "value": ["Bennette", "Campbell"]}]},
                {"type": "arrows", "filters": passes + [{"column": "player", "value": "Bennette"}],
                 "color": "#FFD23F", "size": 1.5, "label": "Pases Bennette"},
                {"type": "scatter", "filters": shots + sap, "marker": "football", "size_by": "xg"},
                {"type": "annotate", "options": {"annotations": [{"text": "Zona de remate", "x": 100, "y": 10,
                                                                  "text_x": 80, "text_y": 5, "box": True}]}},
                {"type": "delaunay", "data": [{"x": 60, "y": 20}, {"x": 70, "y": 60}, {"x": 80, "y": 40},
                                              {"x": 50, "y": 40}], "color": "white"},
                {"type": "polygon", "options": {"vertices": [[102, 18], [120, 18], [120, 62], [102, 62]]},
                 "color": "#FF4F5E", "alpha": 0.15},
                {"type": "text", "data": [{"x": 60, "y": 75, "t": "Texto"}], "label_by": "t"},
                {"type": "plot", "data": [{"x": 20, "y": 10, "o": 1}, {"x": 40, "y": 30, "o": 2},
                                          {"x": 55, "y": 25, "o": 3}], "options": {"sort_by": "o"}, "marker": "o",
                 "color": "#43E8D8", "label": "Recorrido"},
                {"type": "goal_angle", "data": [{"x": 110, "y": 30}], "alpha": 0.2},
                {"type": "zones", "options": {"zones": [[0, 40, 0, 80], [40, 80, 0, 80], [80, 120, 0, 80]],
                                              "names": ["Def", "Med", "Ata"]}, "alpha": 0.4},
                {"type": "formation", "options": {"formation": "442", "half": True, "flip": True}, "size": 200,
                 "color": "#C77DFF"},
            ]}, "pitch_chart_all_layers")
        await call("pitch_chart", {"dataset_id": ev, "layers": [
            {"type": "scatter", "filters": passes, "color_by": "minute", "cmap": "viridis", "size": 40},
            {"type": "hexbin", "filters": sap, "alpha": 0.4},
            {"type": "lines", "filters": passes + [{"column": "player", "value": "Ruiz"}], "color_by": "end_x"},
            {"type": "voronoi", "dataset_id": ff, "group_by": "equipo", "alpha": 0.15},
            {"type": "kde", "filters": shots, "color": "red", "alpha": 0.5},
            {"type": "heatmap_zones", "filters": sap, "options": {"zones": "positional_vertical",
                                                                  "show_labels": True}, "alpha": 0.4},
            {"type": "flow", "filters": passes, "cmap": "Blues"},
            {"type": "sonar", "filters": passes, "options": {"bins": [3, 3], "color_by_stat": "none"},
             "color": "#FF4F5E", "alpha": 0.5},
            {"type": "pass_network", "filters": passes, "group_by": "player",
             "options": {"recipient": "recipient", "label": "initials"}},
            {"type": "comet", "filters": passes + [{"column": "player", "value": "Galo"}]},
        ], "pitch": {"vertical": True, "label": True, "tick": True},
            "figure": {"title": "Todas las capas II", "legend_loc": "below"}}, "pitch_chart_all_layers_2")
        res, txt = await call("transform_dataset", {"dataset_id": ev, "new_name": "opta", "filters": passes,
                                                    "standardize_from": "statsbomb", "standardize_to": "opta",
                                                    "add_angle_distance": True, "pitch_type": "opta"})
        op = json.loads(res.content[0].text)["dataset_id"]
        await call("pass_map", {"dataset_id": op, "pitch": {"pitch_type": "opta"}, "filters": [
            {"column": "player", "value": "Campbell"}], "color_by": "distance"}, "pass_map_opta")
        await call("pass_map", {"dataset_id": op, "pitch": {"pitch_type": "statsbomb"}}, "warn_wrong_pitch")
        # errors should be clean
        await call("shot_map", {"dataset_id": "nope"}, "expected_error_1")
        await call("pass_map", {"dataset_id": ev, "color_by": "colorx"}, "expected_error_2")
        # inline data
        await call("shot_map", {"data": [{"x": 110, "y": 40, "xg": 0.4, "res": "Goal"},
                                         {"x": 100, "y": 30, "xg": 0.1, "res": "Saved"}],
                                "xg": "xg", "outcome": "res"}, "shot_map_inline")
    print("\nFAILED:", FAIL)


asyncio.run(main())
