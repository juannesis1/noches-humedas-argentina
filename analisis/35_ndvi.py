"""Verdor satelital (NDVI) alrededor de las estaciones: ¿bajó en primavera donde se intensificó la agricultura?

NOAA CDR de NDVI (AVHRR y VIIRS, v5; Vermote et al., doi:10.7289/V5ZG6QH9), diario, recortado a 0.1° (32_ndvi_descarga.py).
Para cada estación: compuesto de máximo mensual (filtra nubes) de cada píxel, promedio en un círculo de 100 km y,
por mes (oct-mar), tendencia de Sen 1982-2025. Se relaciona con la expansión de cultivos, con el índice de
intensificación (26_intensificacion.csv) y con la tendencia de Td nocturna del mismo mes.
Expectativa física: si se reemplazaron pasturas perennes por cultivos de verano sembrados en primavera, el NDVI
debería bajar en oct-dic (suelo desnudo o cultivo chico) y no en ene-mar; el trigo de doble cultivo (verde en
octubre) puede compensar en octubre.
Advertencia: el registro mezcla 8 satélites NOAA y VIIRS; la deriva orbital afecta niveles absolutos, por eso la
inferencia se basa en DIFERENCIAS entre estaciones (todas comparten el mismo sensor cada año).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

RADIO = 100


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")].copy()
    est["nombre"] = est.nombre.str.strip()
    archivos = sorted(glob.glob("data/ndvi/ndvi_*.nc"))
    ref = xr.open_dataset(archivos[0])
    lat, lon = ref.latitude.values, ref.longitude.values
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    mascaras = {}
    for _, e in est.iterrows():
        d = np.hypot((LON - e.lon) * 111 * np.cos(np.radians(e.lat)), (LAT - e.lat) * 111)
        mascaras[e.nombre] = d <= RADIO
    # compuesto de máximo mensual por píxel
    por_mes = {}
    for f in archivos:
        ym = re.search(r"ndvi_(\d{6})\d\d\.nc", f).group(1)
        try:
            with xr.open_dataset(f) as ds:
                v = ds.NDVI.values[0].astype(float)
        except Exception:
            continue
        v[(v < -0.2) | (v > 1)] = np.nan
        por_mes[ym] = v if ym not in por_mes else np.fmax(por_mes[ym], v)
    filas = []
    for ym, v in por_mes.items():
        y, m = int(ym[:4]), int(ym[4:])
        for n, mk in mascaras.items():
            x = v[mk]
            if np.isfinite(x).mean() >= 0.5:
                filas.append({"nombre": n, "anio": y, "mes": m, "ndvi": np.nanmean(x)})
    s = pd.DataFrame(filas)
    s["temporada"] = np.where(s.mes >= 10, s.anio + 1, s.anio)
    s.to_csv("analisis/35_ndvi_series.csv", index=False)
    inten = pd.read_csv("analisis/26_intensificacion.csv").set_index("nombre")
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip()).set_index("nombre")
    res = []
    for (n, m), g in s.groupby(["nombre", "mes"]):
        g = g[(g.temporada >= 1982) & (g.temporada <= 2025)].set_index("temporada").ndvi
        if len(g) < 30:
            continue
        res.append({"nombre": n, "mes": m, "tend": sen(g.index.values.astype(float), g.values) * 10, "media": g.mean()})
    r = pd.DataFrame(res)
    r["dcult"] = r.nombre.map(uso.delta_soja_pp)
    r["dint"] = r.nombre.map(inten.d_int)
    r.to_csv("analisis/35_ndvi_tendencias.csv", index=False)
    print("Tendencia de NDVI (por década) por mes; alta (>20 pp) vs baja (<5 pp) expansión; ρ con expansión e intensificación:")
    for m in (10, 11, 12, 1, 2, 3):
        q = r[r.mes == m].dropna(subset=["dcult"])
        a, b = q[q.dcult > 20].tend.mean(), q[q.dcult < 5].tend.mean()
        qi = q.dropna(subset=["dint"])
        print(f"  mes {m:2d}: alta {a:+.3f} baja {b:+.3f} | ρ cultivos {stats.spearmanr(q.dcult, q.tend)[0]:+.2f} "
              f"| ρ intensificación {stats.spearmanr(qi.dint, qi.tend)[0]:+.2f} (n={len(q)})")


if __name__ == "__main__":
    main()
