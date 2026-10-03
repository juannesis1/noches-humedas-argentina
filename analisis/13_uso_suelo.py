"""Hipótesis de uso del suelo (Pal et al. 2021): ¿se secan más las estaciones con más expansión
de soja alrededor?

Fracción de soja = superficie sembrada de soja (MAGyP, por departamento) / superficie, dentro
de un radio de 100 km de cada estación argentina, ponderando cada departamento por la parte
de su área que cae dentro del círculo. Períodos: campañas 1980-84 y 2015-19.
Se correlaciona el cambio de la fracción con la tendencia de Td/HR nocturna (estaciones
homogeneizadas), y se reporta la correlación parcial controlando latitud y longitud, porque
la expansión de la soja tiene estructura geográfica y puede confundirse con el gradiente.
"""
import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats
from shapely.geometry import Point, shape
from shapely.ops import transform
from shapely.validation import make_valid

import os
RADIO_KM = int(os.environ.get("RADIO_KM", 100))
P1, P2 = (1980, 1984), (2015, 2019)


def main():
    fc = json.load(open("data/agro/dep_ign.json"))
    archivos = {"soja": ["soja_1941_2024"], "maiz": ["maiz"], "trigo": ["trigo"], "girasol": ["girasol"],
                "verano": ["soja_1941_2024", "maiz", "girasol"]}[os.environ.get("CULTIVO", "soja")]
    soja = pd.concat([pd.read_csv(f"data/agro/{a}.csv", encoding="latin-1") for a in archivos])
    soja = soja.dropna(subset=["departamento_id"])
    soja["departamento_id"] = pd.to_numeric(soja.departamento_id).astype(int).astype(str).str.zfill(5)
    area = {}
    geoms = {}
    # proyección de igual área para el Cono Sur
    aeq = Transformer.from_crs("EPSG:4326", "+proj=aea +lat_1=-25 +lat_2=-40 +lat_0=-32 +lon_0=-62",
                               always_xy=True).transform
    for f in fc["features"]:
        cod = str(f["properties"]["in1"]).zfill(5)
        g0 = shape(f["geometry"])
        if g0.bounds[1] < -56:          # sector antártico e islas: fuera del dominio
            continue
        try:
            g = make_valid(transform(aeq, g0))
        except Exception:
            continue
        if not np.isfinite(g.area) or g.area <= 0:
            continue
        geoms[cod] = g
        area[cod] = g.area / 1e4          # ha
    s1 = soja[soja.anio.between(*P1)].groupby("departamento_id").superficie_sembrada_ha.sum() / 5
    s2 = soja[soja.anio.between(*P2)].groupby("departamento_id").superficie_sembrada_ha.sum() / 5
    est = pd.read_csv(os.environ.get("TEND", "analisis/03_tendencias_anomalias_laxo_aj.csv"))
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    est = est[est.sid.str.startswith("87")]          # solo Argentina (hay datos de soja)
    filas = []
    for _, e in est.iterrows():
        c = transform(aeq, Point(e.lon, e.lat)).buffer(RADIO_KM * 1000)
        sup = 0.0
        soja1 = soja2 = 0.0
        for cod, g in geoms.items():
            if not g.intersects(c):
                continue
            frac = g.intersection(c).area / g.area
            if not np.isfinite(frac):
                continue
            sup += area[cod] * frac
            soja1 += s1.get(cod, 0.0) * frac
            soja2 += s2.get(cod, 0.0) * frac
        filas.append({"nombre": e.nombre, "lat": e.lat, "lon": e.lon,
                      "soja_80_84_%": 100 * soja1 / sup, "soja_15_19_%": 100 * soja2 / sup,
                      "td_sen": e.td_sen, "hr_sen": e.hr_sen, "t_sen": e.t_sen})
    d = pd.DataFrame(filas)
    d["delta_soja_pp"] = d["soja_15_19_%"] - d["soja_80_84_%"]
    pd.set_option("display.width", 200)
    print(d.round(2).sort_values("delta_soja_pp").to_string(index=False))
    for v in ("td_sen", "hr_sen", "t_sen"):
        r, p = stats.spearmanr(d.delta_soja_pp, d[v])
        # correlación parcial controlando lat y lon (residuos de regresión lineal)
        X = np.c_[np.ones(len(d)), d.lat, d.lon]
        rx = d.delta_soja_pp - X @ np.linalg.lstsq(X, d.delta_soja_pp, rcond=None)[0]
        ry = d[v] - X @ np.linalg.lstsq(X, d[v], rcond=None)[0]
        rp, pp = stats.pearsonr(rx, ry)
        print(f"{v}: Spearman con Δsoja ρ={r:+.2f} (p={p:.3f}); parcial (lat, lon) r={rp:+.2f} (p={pp:.3f}); n={len(d)}")
    d.to_csv(f"analisis/13_uso_suelo_{os.environ.get('CULTIVO', 'soja')}_{RADIO_KM}km.csv", index=False)


if __name__ == "__main__":
    main()
