"""¿La humedad nocturna sigue al verdor primaveral AÑO A AÑO? Y ¿el verdor media la relación cultivos–secado?

1. Panel estación × temporada (20 estaciones argentinas, 1982-2025):
   Td_OND(s, y) = α_s + γ_y + β · NDVI_OND(s, y) [+ δ · lluvia_OND(s, y)] + ε
   NDVI_OND = media de los compuestos de máximo mensual de oct-dic en 100 km (35_ndvi_series.csv); lluvia = CHIRPS
   oct-dic en ±0.5° (desde 1982). γ_y absorbe todo lo común a la región (ENSO, SAM, circulación, sensor satelital de
   cada año). Versión exigente: + tendencia lineal por estación (β identificado solo por desvíos año a año).
   Incertidumbre: bootstrap por estaciones (1000). Si β > 0 (más verde → noches más húmedas) incluso con tendencias
   por estación, la humedad sigue el calendario real de la vegetación, no solo su tendencia.
2. Mediación (transversal): ρ parcial Td~cultivos | lat, lon, tendencia NDVI_OND, y Td~NDVI | lat, lon, cultivos.
Salida: analisis/36_resumen.txt (corto) y CSV con el panel.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

h = importlib.import_module("19_huella_temporal")
rng = np.random.default_rng(21)


def ols_fe(p, cols, tendencia):
    y = p.y.values
    X = [p[cols].values, pd.get_dummies(p.est).values.astype(float),
         pd.get_dummies(p.anio, drop_first=True).values.astype(float)]
    if tendencia:
        X.append(pd.get_dummies(p.est).values.astype(float) * ((p.anio.values - 2000) / 10.0)[:, None])
    return np.linalg.lstsq(np.hstack(X), y, rcond=None)[0][: len(cols)]


def boot(p, cols, tendencia, n=1000):
    ests = p.est.unique()
    g = {e: p[p.est == e] for e in ests}
    out = []
    for _ in range(n):
        el = rng.choice(ests, len(ests))
        q = pd.concat([g[e].assign(est=f"{e}#{i}") for i, e in enumerate(el)])
        out.append(ols_fe(q, cols, tendencia)[0])
    return np.percentile(out, [5, 95])


def parcial(x, y, Z):
    X = np.c_[np.ones(len(x)), Z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def main():
    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")
               & tend.sid.str.startswith("87")].copy()
    est["nombre"] = est.nombre.str.strip()
    nd = pd.read_csv("analisis/35_ndvi_series.csv")
    nd = nd[nd.mes.isin([10, 11, 12])].groupby(["nombre", "temporada"]).ndvi.mean()
    ds = xr.open_dataset("data/chirps/chirps_mensual.nc", decode_times=False)
    meses = pd.date_range("1960-01-01", periods=int(ds.T.max()) + 1, freq="MS")
    pr = ds.precipitation.assign_coords(T=meses[np.floor(ds.T.values).astype(int)])
    h.DIR = "data/noches_ajustadas_pares_laxo"
    filas = []
    for _, e in est.iterrows():
        td = h.serie(e.sid, "td", [10, 11, 12])
        s = pr.sel(Y=slice(e.lat - 0.5, e.lat + 0.5), X=slice(e.lon - 0.5, e.lon + 0.5)).mean(["X", "Y"]).to_series()
        s.index = pd.DatetimeIndex(s.index)
        s = s[s.index.month.isin([10, 11, 12])]
        lluvia = s.groupby(s.index.year + 1).sum()
        for y, v in td.items():
            filas.append({"est": e.nombre, "anio": y, "y": v, "ndvi": nd.get((e.nombre, y), np.nan),
                          "lluvia": lluvia.get(y, np.nan), "lat": e.lat, "lon": e.lon})
    p = pd.DataFrame(filas).dropna()
    p = p[p.anio >= 1982]
    p["ndvi"] = p.ndvi / 0.05                 # β en °C por 0.05 de NDVI
    p["lluvia"] = p.lluvia / 100              # δ en °C por 100 mm
    p.to_csv("analisis/36_ndvi_panel.csv", index=False)
    lineas = [f"Panel: {p.est.nunique()} estaciones, {len(p)} estación-temporadas (oct-dic, 1982-2025)"]
    for cols, lab in ((["ndvi"], "Td ~ NDVI"), (["ndvi", "lluvia"], "Td ~ NDVI + lluvia")):
        for t in (False, True):
            b = ols_fe(p, cols, t)
            ic = boot(p, cols, t)
            lineas.append(f"  {lab:20s} {'+ tend. por estación' if t else 'efectos fijos      '}: β = {b[0]:+.3f} °C por 0.05 NDVI "
                          f"(IC90 {ic[0]:+.3f} a {ic[1]:+.3f})" + (f"; lluvia {b[1]:+.3f} °C/100 mm" if len(b) > 1 else ""))
    # mediación transversal
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip()).set_index("nombre")
    tr = []
    for n, g in p.groupby("est"):
        x = g.anio.values.astype(float)
        tr.append({"nombre": n, "td": sen(x, g.y.values) * 10, "ndvi": sen(x, g.ndvi.values * 0.05) * 10,
                   "lat": g.lat.iloc[0], "lon": g.lon.iloc[0], "cult": uso.delta_soja_pp.get(n, np.nan)})
    q = pd.DataFrame(tr).dropna()
    r0 = parcial(q.cult.values, q.td.values, np.c_[q.lat, q.lon])
    r1 = parcial(q.cult.values, q.td.values, np.c_[q.lat, q.lon, q.ndvi])
    r2 = parcial(q.ndvi.values, q.td.values, np.c_[q.lat, q.lon, q.cult])
    lineas += ["Mediación (tendencias 1982-2025, n=%d):" % len(q),
               f"  Td~cultivos | lat, lon              r = {r0[0]:+.2f} (p {r0[1]:.3f})",
               f"  Td~cultivos | lat, lon, NDVI         r = {r1[0]:+.2f} (p {r1[1]:.3f})",
               f"  Td~NDVI     | lat, lon, cultivos     r = {r2[0]:+.2f} (p {r2[1]:.3f})"]
    open("analisis/36_resumen.txt", "w").write("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
