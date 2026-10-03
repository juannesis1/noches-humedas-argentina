"""Tendencias tras remover ENSO (ONI) y SAM (Marshall 2003) por regresión múltiple.

SAM de temporada = media oct-mar (oct-dic del año anterior). Ambos índices se usan sin
tendencia lineal, para no quitar tendencia climática por colinealidad (el SAM de verano
tiene tendencia positiva desde los 80); se reporta además la versión con SAM con tendencia.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen  # noqa: E402
from importlib import import_module  # noqa: E402

enso = import_module("11_enso")


def sam_temporada():
    s = pd.read_csv("data/indices/sam_marshall.txt", sep=r"\s+", skiprows=2, header=None,
                    names=["anio", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    s = s.set_index("anio")
    out = {}
    for a in s.index:
        if a - 1 in s.index:
            vals = [s.loc[a - 1, m] for m in (10, 11, 12)] + [s.loc[a, m] for m in (1, 2, 3)]
            if np.all(np.isfinite(vals)):
                out[a] = np.mean(vals)
    return pd.Series(out)


def sin_tend(x, anios):
    return x - np.polyval(np.polyfit(anios, x, 1), anios)


def main():
    oni, sam = enso.oni_temporada(), sam_temporada()
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    series = pd.read_csv("analisis/03_series_temporada_anomalias_laxo_aj.csv")
    series = series[series.temp_tipo == "ONDJFM"]
    x_s = sam.reindex(range(1980, 2026)).dropna()
    print("Tendencia del SAM de temporada 1980-2025: %.2f /déc (p=%.3f)" %
          (sen(x_s.index.values.astype(float), x_s.values) * 10,
           mk_hamed_rao(x_s.index.values.astype(float), x_s.values)[1]))
    filas = []
    for _, e in est.iterrows():
        s = series[series.sid == e.sid].set_index("temporada")
        for v in ("td", "tw"):
            y = s[v]
            df = pd.concat([y, oni.rename("oni"), sam.rename("sam")], axis=1).dropna()
            an = df.index.values.astype(float)
            X = np.c_[np.ones(len(df)), sin_tend(df.oni.values, an), sin_tend(df.sam.values, an)]
            b = np.linalg.lstsq(X, df[v].values, rcond=None)[0]
            resid = df[v].values - X[:, 1:] @ b[1:]
            X2 = np.c_[np.ones(len(df)), sin_tend(df.oni.values, an), df.sam.values]
            b2 = np.linalg.lstsq(X2, df[v].values, rcond=None)[0]
            resid2 = df[v].values - X2[:, 1:] @ b2[1:]
            filas.append({"nombre": e.nombre, "lon": e.lon, "var": v,
                          "coef_oni": b[1], "coef_sam": b[2],
                          "sen_bruta": sen(an, df[v].values) * 10,
                          "sen_sin_enso_sam": sen(an, resid) * 10,
                          "p_sin_enso_sam": mk_hamed_rao(an, resid)[1],
                          "sen_sin_enso_sam_contend": sen(an, resid2) * 10})
    res = pd.DataFrame(filas).sort_values(["var", "lon"])
    res.to_csv("analisis/11b_enso_sam.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
