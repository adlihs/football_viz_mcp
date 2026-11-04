"""Generate synthetic sample datasets (StatsBomb 120x80 coordinates) to try the server."""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
out = Path(__file__).parent

home = ["Navas", "Calvo", "Duarte", "Waston", "Oviedo", "Borges", "Galo", "Bennette", "Contreras", "Ruiz", "Campbell"]
away = ["Moreira", "Mitchell", "Vargas", "Quirós", "Matarrita", "Tejeda", "Aguilera", "Zamora", "Brenes", "Venegas", "Ugalde"]
base = {  # rough average positions for a 4-3-3 attacking right
    0: (10, 40), 1: (35, 70), 2: (30, 52), 3: (30, 28), 4: (35, 10), 5: (50, 40),
    6: (62, 55), 7: (62, 25), 8: (85, 70), 9: (85, 10), 10: (95, 40)}
rows = []
for team, squad in (("Saprissa", home), ("Alajuelense", away)):
    for _ in range(520):
        i = rng.integers(0, 11)
        j = rng.choice([k for k in range(11) if k != i and abs(k - i) <= 4])
        sx, sy = base[i]; ex, ey = base[j]
        x0, y0 = np.clip(sx + rng.normal(0, 9), 1, 119), np.clip(sy + rng.normal(0, 8), 1, 79)
        x1, y1 = np.clip(ex + rng.normal(0, 9), 1, 119), np.clip(ey + rng.normal(0, 8), 1, 79)
        ok = rng.random() > (0.12 + 0.2 * (x1 > 90))
        rows.append(dict(team=team, type="Pass", player=squad[i], recipient=squad[j] if ok else None,
                         x=round(x0, 1), y=round(y0, 1), end_x=round(x1, 1), end_y=round(y1, 1),
                         outcome="Complete" if ok else "Incomplete", minute=int(rng.integers(1, 95)),
                         half=1 if rng.random() < .5 else 2))
    for _ in range(16 if team == "Saprissa" else 11):
        i = rng.choice([8, 9, 10, 10, 6, 7])
        x0, y0 = np.clip(rng.normal(104, 7), 85, 119), np.clip(rng.normal(40, 9), 18, 62)
        dist = np.hypot(120 - x0, 40 - y0)
        xg = float(np.clip(np.exp(-dist / 7) * 0.9 + rng.normal(0, .02), 0.01, 0.85))
        res = "Goal" if rng.random() < xg else rng.choice(["Saved", "Off T", "Blocked", "Post"], p=[.4, .35, .2, .05])
        rows.append(dict(team=team, type="Shot", player=squad[i], recipient=None, x=round(x0, 1), y=round(y0, 1),
                         end_x=120, end_y=round(rng.normal(40, 3), 1), outcome=res, minute=int(rng.integers(1, 95)),
                         half=1 if rng.random() < .5 else 2, xg=round(xg, 3)))
    for _ in range(140):  # defensive actions
        x0, y0 = np.clip(rng.normal(45, 20), 1, 119), np.clip(rng.normal(40, 20), 1, 79)
        rows.append(dict(team=team, type=rng.choice(["Pressure", "Tackle", "Interception", "Recovery"]),
                         player=squad[rng.integers(1, 11)], recipient=None, x=round(x0, 1), y=round(y0, 1),
                         end_x=None, end_y=None, outcome=None, minute=int(rng.integers(1, 95)),
                         half=1 if rng.random() < .5 else 2))
pd.DataFrame(rows).to_csv(out / "partido_eventos.csv", index=False)

# player season stats (radar / pizza)
names = home + away + [f"Jugador {k}" for k in range(1, 39)]
st = pd.DataFrame({
    "jugador": names,
    "equipo": ["Saprissa"] * 11 + ["Alajuelense"] * 11 + list(rng.choice(["Herediano", "Cartaginés", "Liberia",
                                                                            "San Carlos"], 38)),
    "posicion": (["POR", "DEF", "DEF", "DEF", "DEF", "MED", "MED", "MED", "DEL", "DEL", "DEL"] * 2
                 + list(rng.choice(["DEF", "MED", "DEL"], 38))),
    "minutos": rng.integers(400, 2700, len(names)),
})
for col, mu, sd in [("goles_90", .25, .18), ("xg_90", .24, .15), ("asist_90", .15, .1), ("pases_clave_90", 1.2, .6),
                    ("regates_90", 1.5, .9), ("pases_prog_90", 4, 2), ("recuperaciones_90", 5, 2),
                    ("duelos_aereos_pct", 50, 12), ("precision_pase_pct", 78, 7), ("perdidas_90", 9, 3)]:
    st[col] = np.round(np.clip(rng.normal(mu, sd, len(names)), 0, None), 2)
st.to_csv(out / "jugadores_temporada.csv", index=False)

# league positions by round (bumpy)
teams = ["Saprissa", "Alajuelense", "Herediano", "Cartaginés", "Liberia", "San Carlos", "Pérez Zeledón",
         "Puntarenas", "Guanacasteca", "Santos", "Sporting", "Municipal Grecia"]
pts = {t: 0 for t in teams}; tab = []
for r in range(1, 15):
    for t in teams:
        pts[t] += int(rng.choice([0, 1, 3], p=[.35, .25, .4]) + (1 if t in teams[:3] and rng.random() < .3 else 0))
    order = sorted(teams, key=lambda t: -pts[t])
    tab += [dict(jornada=r, equipo=t, puntos=pts[t], posicion=order.index(t) + 1) for t in teams]
pd.DataFrame(tab).to_csv(out / "tabla_por_jornada.csv", index=False)

# freeze frame of a shot
ff = [dict(jugador=n, equipo="Saprissa", x=round(float(x), 1), y=round(float(y), 1)) for n, (x, y) in
      zip(home[5:], [(80, 40), (92, 60), (95, 22), (104, 66), (101, 14), (108, 44)])]
ff += [dict(jugador=n, equipo="Alajuelense", x=round(float(x), 1), y=round(float(y), 1)) for n, (x, y) in
       zip(away, [(118, 40), (110, 50), (111, 36), (109, 30), (106, 58), (100, 44), (98, 30), (96, 52),
                  (90, 40), (86, 60), (70, 35)])]
pd.DataFrame(ff).to_csv(out / "freeze_frame_tiro.csv", index=False)
print("ok")
