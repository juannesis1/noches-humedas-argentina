"""Índices de estrés térmico con uso sanitario, aplicados a la noche (06+09 UTC).

- Heat index de la NWS (Rothfusz 1990 con los ajustes de Steadman/NWS para HI < 80 °F y
  para humedad alta/baja), en °C.
- sWBGT simplificado del Australian Bureau of Meteorology: 0.567 T + 0.393 e + 3.94
  (e en hPa). Es la forma usada en la literatura de salud (p. ej. Willett & Sherwood 2012).
Ambos se calculan sobre la media nocturna de T y Td (aproximación: las dos horas se
promedian antes de aplicar el índice).
Métricas por temporada ONDJFM: media de anomalías (clim. diaria 1981-2010, igual que en
03_tendencias), % de noches por encima del P90 propio de la estación (1981-2010) y % de
noches "sin alivio" con HI ≥ 25 °C. Mismo criterio de temporadas válidas y estaciones que el
criterio laxo (≥30 temporadas válidas, ≥5 en cada extremo).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import fdr, mk_hamed_rao, sen  # noqa: E402
import termo  # noqa: E402

DIR = os.environ.get("DIR_NOCHES", "data/noches_ajustadas_pares_laxo")
MESES = [10, 11, 12, 1, 2, 3]
UMBRAL_HI = 25.0


def heat_index(t_c, rh):
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = -42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh \
        - 6.83783e-3 * t ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh \
        + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2
    seco = (rh < 13) & (t >= 80) & (t <= 112)
    with np.errstate(invalid="ignore"):
        hi = np.where(seco, hi - (13 - rh) / 4 * np.sqrt((17 - np.abs(t - 95)) / 17), hi)
    humedo = (rh > 85) & (t >= 80) & (t <= 87)
    hi = np.where(humedo, hi + (rh - 85) / 10 * (87 - t) / 5, hi)
    hi = np.where((simple + t) / 2 < 80, simple, hi)
    return (hi - 32) * 5 / 9


def swbgt(t_c, td_c):
    return 0.567 * t_c + 0.393 * termo.es_bolton(td_c) + 3.94


def anomalia(d, v):
    base = d[(d.temporada >= 1981) & (d.temporada <= 2010)]
    clim = base[v].groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True,
                                                                      min_periods=10).mean()
    clim = clim.iloc[15:-15]
    clim.index = range(1, 367)
    return d[v] - clim.loc[d.index.dayofyear].values


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    filas = []
    for f in sorted(glob.glob(f"{DIR}/*.parquet")):
        sid = os.path.basename(f)[:-8]
        if "CERES" in info.loc[sid, "nombre"]:      # >3 cortes: excluida (04b)
            continue
        d = pd.read_parquet(f)
        d = d[d.index.month.isin(MESES)].copy()
        d["hi"] = heat_index(d.t.values, termo.rh_desde_td(d.t.values, d.td.values))
        d["swbgt"] = swbgt(d.t.values, d.td.values)
        base = d[(d.temporada >= 1981) & (d.temporada <= 2010)]
        p90 = {v: np.nanpercentile(base[v], 90) for v in ("hi", "swbgt")}
        for v in ("hi", "swbgt"):
            d[f"a_{v}"] = anomalia(d, v)
            d[f"p90_{v}"] = (d[v] >= p90[v]) * 100.0
        d["sin_alivio"] = (d.hi >= UMBRAL_HI) * 100.0
        g = d.groupby("temporada")
        cols = ["a_hi", "a_swbgt", "p90_hi", "p90_swbgt", "sin_alivio"]
        med = g[cols].mean()
        pm = d.groupby(["temporada", d.index.month]).size().unstack().reindex(columns=MESES)
        ok = (g.size() >= 91) & (pm.fillna(0) >= 9).all(axis=1)
        med = med[ok & (med.index >= 1980) & (med.index <= 2025)]
        incl = len(med) >= 30 and (med.index <= 1989).sum() >= 5 and (med.index >= 2016).sum() >= 5
        if not incl:
            continue
        x = med.index.values.astype(float)
        fila = {"sid": sid, "nombre": info.loc[sid, "nombre"], "lat": info.loc[sid, "lat"],
                "lon": info.loc[sid, "lon"], "n": len(med),
                "hi_clim": base.hi.mean(), "sin_alivio_clim": (base.hi >= UMBRAL_HI).mean() * 100}
        for c in cols:
            fila[f"{c}_sen"] = sen(x, med[c].values) * 10
            fila[f"{c}_p"] = mk_hamed_rao(x, med[c].values)[1]
        filas.append(fila)
    res = pd.DataFrame(filas)
    for c in ["a_hi", "a_swbgt", "p90_hi", "p90_swbgt", "sin_alivio"]:
        res[f"{c}_fdr"] = fdr(res[f"{c}_p"].values)
    res.to_csv("analisis/17_indice_salud.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.sort_values("lat")[["nombre", "lat", "hi_clim", "a_hi_sen", "a_swbgt_sen", "p90_hi_sen",
                                  "p90_swbgt_sen", "sin_alivio_clim", "sin_alivio_sen",
                                  "a_hi_fdr", "p90_hi_fdr", "sin_alivio_fdr"]].round(2).to_string(index=False))
    print(f"\nn = {len(res)} estaciones (medianas por década)")
    for c, u in (("a_hi", "°C"), ("a_swbgt", "°C"), ("p90_hi", "pp"), ("p90_swbgt", "pp"), ("sin_alivio", "pp")):
        s = res[f"{c}_sen"]
        print(f"{c:11s}: mediana {s.median():+.3f} {u}/déc, positivas {(s > 0).mean() * 100:.0f} %, "
              f"significativas FDR {(res[f'{c}_fdr']).sum()}")


if __name__ == "__main__":
    main()
