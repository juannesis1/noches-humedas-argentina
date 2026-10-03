"""¿La vegetación primaveral resume el estado de la superficie mejor que la humedad del suelo? (revisión I-A)

La lluvia CHIRPS es un control ruidoso; la humedad superficial del suelo satelital (ESA CCI SM v09.2 COMBINED) es el
control directo del agua disponible en la superficie. Para cada estación (23), media de oct-dic en 100 km de las
anomalías de humedad del suelo por celda (cada celda respecto de su propia media 1991-2020, para que el cambio de
cobertura de sensores no altere la composición de la media), temporadas 1982-2025.
  (a) Transversal: ρ(Td, SM), parciales Td~NDVI | SM y Td~SM | NDVI, y Td~cultivos | SM (estaciones argentinas).
  (b) Panel año a año: Td ~ NDVI + SM con efectos fijos de estación y año y tendencias por estación (bootstrap).
  (c) Sensibilidad: solo desde 1992 (ERS en adelante; antes la cobertura es escasa).
Salida: analisis/40_resumen.txt
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
p36 = importlib.import_module("36_ndvi_panel")
nul = importlib.import_module("24_nulo_espacial")


def sm_estaciones(est):
    ds = xr.open_mfdataset("data/esacci_sm/sm_[0-9]*_*.nc", combine="by_coords").load()
    sm = ds.sm.where(ds.n >= 5)
    clim = sm.sel(time=slice("1990-10-01", "2020-12-31")).mean("time")
    an = sm - clim
    temp = an.time.dt.year.values + 1                      # oct-dic de y-1 → temporada y
    an = an.assign_coords(temporada=("time", temp)).groupby("temporada").mean()
    lat2, lon2 = np.meshgrid(an.lat.values, an.lon.values, indexing="ij")
    out = {}
    for _, e in est.iterrows():
        d = nul.dist(np.r_[e.lat, lat2.ravel()], np.r_[e.lon, lon2.ravel()])[0, 1:].reshape(lat2.shape)
        m = xr.DataArray(d <= 100, dims=("lat", "lon"))
        v = an.where(m)
        ok = v.notnull().sum(["lat", "lon"]) >= 5
        out[e.est] = v.mean(["lat", "lon"]).where(ok).to_series()
    return pd.DataFrame(out)


def main():
    t = pd.read_csv("analisis/37_resumen_estaciones.csv")
    sm = sm_estaciones(t)
    L = [f"ESA CCI SM: temporadas {sm.index.min()}-{sm.index.max()}, estaciones {sm.shape[1]}"]
    for ini in (1982, 1992):
        s = sm.loc[ini:2025]
        t[f"sm{ini}"] = [sen(s[n].dropna().index.values.astype(float), s[n].dropna().values) * 10
                         if s[n].notna().sum() >= 15 else np.nan for n in t.est]
    for ini in (1982, 1992):
        q = t.dropna(subset=[f"sm{ini}"])
        x = q[f"sm{ini}"].values
        r_td = stats.spearmanr(q.td, x)[0]
        r_nd = stats.spearmanr(q.ndvi, x)[0]
        a = p36.parcial(q.ndvi.values, q.td.values, np.c_[x])
        b = p36.parcial(x, q.td.values, np.c_[q.ndvi])
        L.append(f"(a) desde {ini}, n {len(q)}: ρ(Td, SM) {r_td:+.2f}; ρ(NDVI, SM) {r_nd:+.2f}; "
                 f"Td~NDVI | SM r {a[0]:+.2f} (p {a[1]:.3f}); Td~SM | NDVI r {b[0]:+.2f} (p {b[1]:.3f})")
        ar = q.dropna(subset=["cult"])
        c0 = p36.parcial(ar.cult.values, ar.td.values, np.c_[ar.lat, ar.lon])
        c1 = p36.parcial(ar.cult.values, ar.td.values, np.c_[ar.lat, ar.lon, ar[f"sm{ini}"]])
        c2 = stats.spearmanr(ar.cult, ar[f"sm{ini}"])
        L.append(f"    Argentina n {len(ar)}: Td~cultivos | lat,lon r {c0[0]:+.2f}; | lat,lon,SM r {c1[0]:+.2f} "
                 f"(p {c1[1]:.3f}); ρ(cultivos, tendencia SM) {c2[0]:+.2f} (p {c2[1]:.3f})")
    # (b) panel año a año
    h.DIR = "data/noches_ajustadas_pares_laxo"
    tr = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    tr = tr[(tr.temporada == "ONDJFM") & tr.incluida & ~tr.nombre.str.contains("CERES")].copy()
    tr["nombre"] = tr.nombre.str.strip()
    nd = pd.read_csv("analisis/35_ndvi_series.csv")
    nd = nd[nd.mes.isin([10, 11, 12])].groupby(["nombre", "temporada"]).ndvi.mean()
    filas = []
    for _, e in tr.iterrows():
        for y, v in h.serie(e.sid, "td", [10, 11, 12]).items():
            filas.append({"est": e.nombre, "anio": y, "y": v, "ndvi": nd.get((e.nombre, y), np.nan) / 0.05,
                          "sm": sm[e.nombre].get(y, np.nan) / 0.01 if e.nombre in sm else np.nan})
    p = pd.DataFrame(filas).dropna()
    p = p[p.anio >= 1982]
    for cols in (["sm"], ["ndvi", "sm"]):
        bb = p36.ols_fe(p, cols, True)
        ic = p36.boot(p, cols, True, n=500)
        L.append(f"(b) panel {len(p)} est-temp, tend. por estación: Td~{'+'.join(cols)}: β1 {bb[0]:+.3f} "
                 f"(IC90 {ic[0]:+.3f} a {ic[1]:+.3f})" + (f"; β_sm {bb[1]:+.3f}" if len(bb) > 1 else "")
                 + "  [°C por 0.05 NDVI / por 0.01 m³ m⁻³]")
    t.to_csv("analisis/40_estaciones.csv", index=False)
    open("analisis/40_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
