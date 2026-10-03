"""Huella temporal del uso del suelo: ¿la Td de cada estación sigue el CALENDARIO de su expansión?

Hasta 13/14 la evidencia es espacial (dónde se expandió más, se secó más). Acá se usa el
tiempo: panel estación × temporada, 1980-2025, con efectos fijos de estación y de año:
    Td_OND(s, y) = α_s + γ_y + β · cultivo(s, y) + ε
γ_y absorbe todo lo común a la región (ENSO, SAM, jet, circulación); α_s, el nivel de cada
estación. β se identifica solo con las diferencias en el momento de la expansión entre zonas.
Versión exigente: + tendencia lineal propia de cada estación (β queda identificado por la forma
no lineal de la expansión, p. ej. el salto de la soja 1995-2010).
cultivo(s, y) = % de la superficie en 100 km sembrado con cultivos de verano en la campaña
que empieza en oct del año y-1 (temporada y = oct y-1 a mar y), igual ponderación por área que 13.
Incertidumbre: bootstrap por estaciones (1000 réplicas; las estaciones son los conglomerados).
Placebos: trigo (cultivo de invierno), ventana JFM y T nocturna.
"""
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform
from shapely.validation import make_valid

RADIO_KM = 100
DIR = "data/noches_ajustadas_pares_laxo"
VENTANAS = {"OND": [10, 11, 12], "JFM": [1, 2, 3], "ONDJFM": [10, 11, 12, 1, 2, 3]}
CULTIVOS = {"verano": ["soja_1941_2024", "maiz", "girasol"], "trigo": ["trigo"]}
rng = np.random.default_rng(42)


def pesos(est):
    fc = json.load(open("data/agro/dep_ign.json"))
    aeq = Transformer.from_crs("EPSG:4326", "+proj=aea +lat_1=-25 +lat_2=-40 +lat_0=-32 +lon_0=-62",
                               always_xy=True).transform
    geoms = {}
    for f in fc["features"]:
        g0 = shape(f["geometry"])
        if g0.bounds[1] < -56:
            continue
        try:
            g = make_valid(transform(aeq, g0))
        except Exception:
            continue
        if np.isfinite(g.area) and g.area > 0:
            geoms[str(f["properties"]["in1"]).zfill(5)] = g
    W, sup = {}, {}
    for _, e in est.iterrows():
        c = transform(aeq, Point(e.lon, e.lat)).buffer(RADIO_KM * 1000)
        w, s = {}, 0.0
        for cod, g in geoms.items():
            if g.intersects(c):
                fr = g.intersection(c).area / g.area
                if np.isfinite(fr):
                    w[cod] = fr
                    s += g.area / 1e4 * fr
        W[e.nombre], sup[e.nombre] = w, s
    return W, sup


def fraccion_anual(est, W, sup, archivos):
    c = pd.concat([pd.read_csv(f"data/agro/{a}.csv", encoding="latin-1") for a in archivos])
    c = c.dropna(subset=["departamento_id"])
    c["departamento_id"] = pd.to_numeric(c.departamento_id).astype(int).astype(str).str.zfill(5)
    tab = c.groupby(["anio", "departamento_id"]).superficie_sembrada_ha.sum().unstack(fill_value=0)
    out = {}
    for n in est.nombre:
        w = pd.Series(W[n]).reindex(tab.columns).fillna(0)
        out[n] = tab.values @ w.values / sup[n] * 100
    f = pd.DataFrame(out, index=tab.index)
    f.index = f.index + 1          # campaña que empieza en oct de 'anio' → temporada anio+1
    return f


def anomalias(d, v):
    base = d[(d.temporada >= 1981) & (d.temporada <= 2010)]
    clim = base[v].groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True,
                                                                      min_periods=10).mean()
    clim = clim.iloc[15:-15]
    clim.index = range(1, 367)
    return d[v] - clim.loc[d.index.dayofyear].values


def serie(sid, v, meses):
    d = pd.read_parquet(f"{DIR}/{sid}.parquet")
    d = d[d.index.month.isin(meses)].copy()
    d["a"] = anomalias(d, v)
    g = d.groupby("temporada")
    pm = d.groupby(["temporada", d.index.month]).size().unstack().reindex(columns=meses)
    ok = (g.size() >= 0.5 * 30.5 * len(meses)) & (pm.fillna(0) >= 9).all(axis=1)
    s = g.a.mean()[ok]
    return s[(s.index >= 1980) & (s.index <= 2025)]


def efectos_fijos(panel, tendencia=False):
    """β por mínimos cuadrados con efectos fijos de estación y año (dummies)."""
    y = panel.y.values
    X = [panel.x.values[:, None],
         pd.get_dummies(panel.est, drop_first=False).values.astype(float),
         pd.get_dummies(panel.anio, drop_first=True).values.astype(float)]
    if tendencia:
        t = (panel.anio.values - 2000) / 10.0
        X.append(pd.get_dummies(panel.est).values.astype(float) * t[:, None])
    X = np.hstack(X)
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


def bootstrap(panel, tendencia, n=1000):
    ests = panel.est.unique()
    grupos = {e: panel[panel.est == e] for e in ests}
    bs = []
    for _ in range(n):
        elegidas = rng.choice(ests, len(ests), replace=True)
        partes = []
        for i, e in enumerate(elegidas):
            p = grupos[e].copy()
            p["est"] = f"{e}#{i}"
            partes.append(p)
        bs.append(efectos_fijos(pd.concat(partes), tendencia))
    return np.percentile(bs, [5, 95]), np.mean(np.array(bs) <= 0)


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"])
    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")
               & tend.sid.str.startswith("87")][["sid", "nombre", "lat", "lon"]]
    W, sup = pesos(est)
    fr = {k: fraccion_anual(est, W, sup, a) for k, a in CULTIVOS.items()}
    fr["verano"].to_csv("analisis/19_fraccion_anual_verano.csv")
    filas = []
    for cult in ("verano", "trigo"):
        for var in ("td", "t"):
            for vn, meses in VENTANAS.items():
                if cult == "trigo" and vn != "OND":
                    continue
                partes = []
                for _, e in est.iterrows():
                    s = serie(e.sid, var, meses)
                    x = fr[cult][e.nombre].reindex(s.index)
                    partes.append(pd.DataFrame({"est": e.nombre, "anio": s.index, "y": s.values,
                                                "x": x.values}))
                panel = pd.concat(partes).dropna()
                for tendencia in (False, True):
                    b = efectos_fijos(panel, tendencia)
                    ic, p1 = bootstrap(panel, tendencia, n=500)
                    fila = {"cultivo": cult, "var": var, "ventana": vn, "tend_est": tendencia,
                            "beta_por_10pp": b * 10, "ic90_lo": ic[0] * 10, "ic90_hi": ic[1] * 10,
                            "frac_boot_<=0": p1, "n": len(panel), "n_est": panel.est.nunique()}
                    filas.append(fila)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()},
                          flush=True)
    pd.DataFrame(filas).to_csv("analisis/19_huella_temporal.csv", index=False)


if __name__ == "__main__":
    main()
