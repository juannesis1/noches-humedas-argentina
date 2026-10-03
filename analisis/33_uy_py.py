"""Cambio de cobertura agrícola alrededor de las estaciones de Uruguay y Paraguay (no hay datos de siembra).

Uruguay: MapBiomas Uruguay Col. 3, tabla de áreas por departamento (Carrasco ≈ Montevideo + Canelones + San José;
Rocha ≈ Rocha + Maldonado + Lavalleja). Paraguay: MapBiomas Paraguay Col. 2, círculo de 100 km de Asunción (solo
territorio paraguayo). Clase 18 = agricultura, 15 = pastura; fracción del área terrestre, 1985-89 vs 2015-19.
Sirve para ubicar a estas 3 estaciones en el gradiente de cambio agrícola, sin índice de intensificación.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

c = pd.read_excel("data/mapbiomas_uy/estadisticas_col3.xlsx", "COVERAGE")
grupos = {"CARRASCO INTL": ["MONTEVIDEO", "CANELONES", "SAN JOSE"], "ROCHA": ["ROCHA", "MALDONADO", "LAVALLEJA"]}
filas = []
for est, deps in grupos.items():
    s = c[c.territory_level_2.isin(deps)]
    tierra = s[s["class"] != 33]
    for clase, nom in ((18, "agricultura"), (15, "pastura")):
        f = lambda y0, y1: s[s["class"] == clase][[f"y{y}" for y in range(y0, y1 + 1)]].sum().mean() / \
            tierra[[f"y{y}" for y in range(y0, y1 + 1)]].sum().mean() * 100
        filas.append({"estacion": est, "clase": nom, "1985-89": f(1985, 1989), "2015-19": f(2015, 2019)})
lat, lon = -25.24, -57.52
for clase, nom in ((18, "agricultura"), (15, "pastura")):
    vals = {}
    for per, anios in (("1985-89", range(1985, 1990)), ("2015-19", range(2015, 2020))):
        fr = []
        for y in anios:
            with rasterio.open(f"data/mapbiomas_py/py_{y}.tif") as r:
                d = 100 / 111
                w = from_bounds(lon - d / np.cos(np.radians(lat)), lat - d, lon + d / np.cos(np.radians(lat)), lat + d, r.transform)
                a = r.read(1, window=w, out_shape=(500, 500), boundless=True, fill_value=0)
            yy, xx = np.mgrid[0:500, 0:500]
            dist = np.hypot((xx - 249.5) / 249.5, (yy - 249.5) / 249.5)
            v = a[(dist <= 1) & (a != 0) & (a != 33) & (a != 26)]
            fr.append(np.mean(v == clase) * 100)
        vals[per] = np.mean(fr)
    filas.append({"estacion": "SILVIO PETTIROSSI INTL (Asunción)", "clase": nom, **vals})
r = pd.DataFrame(filas)
r["cambio"] = r["2015-19"] - r["1985-89"]
r.to_csv("analisis/33_uy_py.csv", index=False)
print(r.round(1).to_string(index=False))
