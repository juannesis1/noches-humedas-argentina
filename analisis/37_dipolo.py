"""¿Un mismo proceso explica los dos polos del dipolo? Vegetación primaveral y humedad nocturna en todas las estaciones.

Hipótesis: el dipolo de Td de oct-dic (litoral NE se humedece, Pampa intensificada se seca) refleja cambios opuestos
de la superficie en primavera: más lluvia y más verdor en el NE; intensificación agrícola que frenó el verdor en la Pampa.
Si es así:
 (a) en TODAS las estaciones (23, incluidas Uruguay y Paraguay), la tendencia de Td oct-dic debe seguir a la de NDVI oct-dic;
 (b) la relación año a año Td~NDVI (efectos fijos de estación y año, con tendencias por estación y lluvia) debe valer
     también en el grupo de BAJA expansión, no solo en la Pampa;
 (c) el NDVI debe "explicar" el gradiente espacial (dipolo): Td ~ lon + lat pierde fuerza al controlar por NDVI.
Lluvia: CHIRPS oct-dic en ±0.5°. Salida: analisis/37_resumen.txt.
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


def main():
    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")].copy()
    est["nombre"] = est.nombre.str.strip()
    nd = pd.read_csv("analisis/35_ndvi_series.csv")
    nd = nd[nd.mes.isin([10, 11, 12])].groupby(["nombre", "temporada"]).ndvi.mean()
    ds = xr.open_dataset("data/chirps/chirps_mensual.nc", decode_times=False)
    meses = pd.date_range("1960-01-01", periods=int(ds.T.max()) + 1, freq="MS")
    pr = ds.precipitation.assign_coords(T=meses[np.floor(ds.T.values).astype(int)])
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip()).set_index("nombre")
    h.DIR = "data/noches_ajustadas_pares_laxo"
    filas = []
    for _, e in est.iterrows():
        td = h.serie(e.sid, "td", [10, 11, 12])
        s = pr.sel(Y=slice(e.lat - 0.5, e.lat + 0.5), X=slice(e.lon - 0.5, e.lon + 0.5)).mean(["X", "Y"]).to_series()
        s.index = pd.DatetimeIndex(s.index)
        s = s[s.index.month.isin([10, 11, 12])]
        ll = s.groupby(s.index.year + 1).sum()
        for y, v in td.items():
            filas.append({"est": e.nombre, "anio": y, "y": v, "ndvi": nd.get((e.nombre, y), np.nan) / 0.05,
                          "lluvia": ll.get(y, np.nan) / 100, "lat": e.lat, "lon": e.lon,
                          "cult": uso.delta_soja_pp.get(e.nombre, np.nan)})
    p = pd.DataFrame(filas).dropna(subset=["y", "ndvi", "lluvia"])
    p = p[p.anio >= 1982]
    L = [f"Estaciones: {p.est.nunique()} (todas, incl. Uruguay y Paraguay); estación-temporadas: {len(p)}"]
    # (a) y (c) transversal
    tr = []
    for n, g in p.groupby("est"):
        x = g.anio.values.astype(float)
        tr.append({"est": n, "td": sen(x, g.y.values) * 10, "ndvi": sen(x, g.ndvi.values * 0.05) * 10,
                   "lluvia": sen(x, g.lluvia.values * 100) * 10, "lat": g.lat.iloc[0], "lon": g.lon.iloc[0],
                   "cult": g.cult.iloc[0]})
    t = pd.DataFrame(tr)
    t.to_csv("analisis/37_resumen_estaciones.csv", index=False)
    L.append(f"(a) Tendencias oct-dic, 23 estaciones: ρ(Td, NDVI) = {stats.spearmanr(t.td, t.ndvi)[0]:+.2f}; "
             f"ρ(Td, lluvia) = {stats.spearmanr(t.td, t.lluvia)[0]:+.2f}; ρ(NDVI, lluvia) = {stats.spearmanr(t.ndvi, t.lluvia)[0]:+.2f}")
    X0 = np.c_[np.ones(len(t)), t.lon, t.lat]
    r2 = lambda X: 1 - np.sum((t.td - X @ np.linalg.lstsq(X, t.td, rcond=None)[0]) ** 2) / np.sum((t.td - t.td.mean()) ** 2)
    L.append(f"(c) Varianza espacial de la tendencia de Td explicada: lon+lat {r2(X0):.2f}; NDVI solo {r2(np.c_[np.ones(len(t)), t.ndvi]):.2f}; "
             f"lon+lat+NDVI {r2(np.c_[X0, t.ndvi]):.2f}; NDVI+lluvia {r2(np.c_[np.ones(len(t)), t.ndvi, t.lluvia]):.2f}")
    r_ll = p36.parcial(t.lon.values, t.td.values, np.c_[t.lat])[0]
    r_ll_nd = p36.parcial(t.lon.values, t.td.values, np.c_[t.lat, t.ndvi])[0]
    L.append(f"    Gradiente E-O del dipolo (Td~lon | lat): r = {r_ll:+.2f}; controlando NDVI: r = {r_ll_nd:+.2f}")
    # (b) panel por grupos
    grupos = {"todas (23)": p, "baja expansión (<5 pp, + UY/PY)": p[(p.cult < 5) | p.cult.isna()],
              "alta expansión (>20 pp)": p[p.cult > 20]}
    for lab, q in grupos.items():
        for cols, t_ in ((["ndvi"], True), (["ndvi", "lluvia"], True)):
            b = p36.ols_fe(q, cols, t_)
            ic = p36.boot(q, cols, t_, n=500)
            L.append(f"(b) {lab:32s} Td~{'+'.join(cols):12s} con tend. por estación: β {b[0]:+.3f} °C/0.05 NDVI "
                     f"(IC90 {ic[0]:+.3f} a {ic[1]:+.3f}), n est {q.est.nunique()}")
    open("analisis/37_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
