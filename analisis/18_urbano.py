"""Sensibilidad a la urbanización con una métrica objetiva (GHSL, superficie construida).

GHS-BUILT-S R2023A (Pesaresi & Politis 2023, JRC), grilla de 1 km en Mollweide: m² construidos
por celda en 1975 y 2020. Para cada estación: fracción construida media en radios de 3 y 10 km
y su cambio 1975-2020. Reemplaza la lista de estaciones "urbanas" elegida a mano.
Pruebas:
1. ¿La urbanización explica el calentamiento? Spearman entre Δconstruido y las tendencias de T
   nocturna, diurna y noche − día (datos crudos en ambos para que la diferencia sea comparable;
   el efecto urbano típico es mayor de noche).
2. Calentamiento "rural": ordenada al origen de la regresión tendencia T ~ Δconstruido.
3. ¿La urbanización confunde el resultado agrícola? Correlación parcial Td ~ Δcultivos
   controlando lat, lon y Δconstruido (ONDJFM y oct-dic).
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

GHSL = "data/ghsl/GHS_BUILT_S_E{}_GLOBE_R2023A_54009_1000_V1_0.tif"
RADIOS = (3, 10)


def construido(lat, lon, anio, radio_km):
    with rasterio.open(GHSL.format(anio)) as r:
        x, y = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(lon, lat)
        fila, col = r.index(x, y)
        n = radio_km + 1
        ventana = rasterio.windows.Window(col - n, fila - n, 2 * n + 1, 2 * n + 1)
        a = r.read(1, window=ventana).astype(float)
        a[a == r.nodata] = np.nan
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    dentro = np.hypot(yy, xx) <= radio_km
    return np.nanmean(a[dentro]) / 1e6 * 100     # % de superficie construida


def parcial(x, y, Z):
    X = np.c_[np.ones(len(x)), Z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def main():
    def leer(f):
        d = pd.read_csv(f)
        d = d[(d.temporada == "ONDJFM") & d.incluida & ~d.nombre.str.contains("CERES")]
        d["nombre"] = d.nombre.str.strip()
        return d.set_index("nombre")
    aj = leer("analisis/03_tendencias_anomalias_laxo_aj.csv")
    crudo = leer("analisis/03_tendencias_anomalias_laxo.csv")
    dia = leer("analisis/03_tendencias_anomalias_dia_laxo.csv")
    dia_aj = leer("analisis/03_tendencias_anomalias_dia_laxo_aj.csv")   # mismos quiebres, ajuste diurno
    est = aj[["lat", "lon", "t_sen", "td_sen", "hr_sen"]].copy()
    est["t_noche_crudo"] = crudo.t_sen
    est["t_dia_crudo"] = dia.t_sen.reindex(est.index)
    est["noche_menos_dia"] = est.t_noche_crudo - est.t_dia_crudo
    est["t_dia_aj"] = dia_aj.t_sen.reindex(est.index)
    est["noche_menos_dia_aj"] = est.t_sen - est.t_dia_aj
    for rk in RADIOS:
        for anio in (1975, 2020):
            est[f"b{anio}_{rk}km"] = [construido(r.lat, r.lon, anio, rk) for r in est.itertuples()]
        est[f"db_{rk}km"] = est[f"b2020_{rk}km"] - est[f"b1975_{rk}km"]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv")
    uso["nombre"] = uso.nombre.str.strip()
    est = est.join(uso.set_index("nombre")[["delta_soja_pp"]])
    fl = pd.read_csv("analisis/16_flujo_local.csv")
    fl = fl[(fl.ventana == "OND") & fl.calma].set_index("nombre")
    est["td_ond"] = fl.tend_total.reindex(est.index)
    est.to_csv("analisis/18_urbano.csv")
    pd.set_option("display.width", 250)
    print(est.sort_values("db_10km", ascending=False)[
        ["b1975_3km", "b2020_3km", "db_3km", "b1975_10km", "b2020_10km", "db_10km",
         "t_sen", "t_dia_aj", "noche_menos_dia_aj", "t_noche_crudo", "t_dia_crudo", "td_sen", "delta_soja_pp"]
    ].round(2).to_string())

    print("\n1. Urbanización vs tendencias (Spearman, n=%d)" % len(est))
    for rk in RADIOS:
        for c in ("t_sen", "t_dia_aj", "noche_menos_dia_aj", "t_noche_crudo", "t_dia_crudo",
                  "noche_menos_dia", "td_sen", "hr_sen"):
            s = est[[f"db_{rk}km", c]].dropna()
            rho, p = stats.spearmanr(s.iloc[:, 0], s.iloc[:, 1])
            print(f"  Δconstruido {rk:2d} km vs {c:16s}: ρ {rho:+.2f} (p {p:.3f}, n {len(s)})")

    print("\n2. Calentamiento nocturno 'rural' (ordenada al origen, T ajustada ~ Δconstruido)")
    for rk in RADIOS:
        res = stats.theilslopes(est.t_sen, est[f"db_{rk}km"])
        ols = stats.linregress(est[f"db_{rk}km"], est.t_sen)
        print(f"  {rk:2d} km: Theil-Sen ordenada {res.intercept:+.3f} °C/déc, pendiente {res.slope:+.4f} por pp; "
              f"OLS ordenada {ols.intercept:+.3f} ± {ols.intercept_stderr:.3f}; mediana de todas {est.t_sen.median():+.3f}")

    print("\n2b. Parcial (lat, lon) T ~ Δconstruido 10 km, y sacando de a una estación")
    for c in ("t_sen", "t_dia_aj", "noche_menos_dia_aj", "t_noche_crudo"):
        r, p = parcial(est.db_10km.values, est[c].values, np.c_[est.lat, est.lon])
        jk = [(n, parcial(est.drop(n).db_10km.values, est.drop(n)[c].values,
                          np.c_[est.drop(n).lat, est.drop(n).lon])[0]) for n in est.index]
        mn = min(jk, key=lambda t: abs(t[1]))
        print(f"  {c:18s}: {r:+.2f} (p {p:.3f}); jackknife {min(j[1] for j in jk):+.2f} a "
              f"{max(j[1] for j in jk):+.2f} (más débil sin {mn[0]})")

    print("\n3. Resultado agrícola controlando urbanización")
    s = est.dropna(subset=["delta_soja_pp"])
    for nombre, col in (("ONDJFM", "td_sen"), ("OND", "td_ond")):
        ss = s.dropna(subset=[col])
        r0, p0 = parcial(ss.delta_soja_pp.values, ss[col].values, np.c_[ss.lat, ss.lon])
        print(f"  {nombre}: parcial lat/lon {r0:+.2f} (p {p0:.3f}, n {len(ss)})")
        for rk in RADIOS:
            r, p = parcial(ss.delta_soja_pp.values, ss[col].values,
                           np.c_[ss.lat, ss.lon, ss[f"db_{rk}km"]])
            r2, p2 = parcial(ss.delta_soja_pp.values, ss[col].values,
                             np.c_[ss.lat, ss.lon, ss[f"b2020_{rk}km"]])
            print(f"     + Δconstruido {rk} km: {r:+.2f} (p {p:.3f});  + construido 2020 {rk} km: {r2:+.2f} (p {p2:.3f})")


if __name__ == "__main__":
    main()
