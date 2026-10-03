"""¿Las tendencias nocturnas sobreviven al remover ENSO?

ONI de la temporada cálida = media de los trimestres OND, NDJ, DJF, JFM (NOAA CPC).
Para cada estación y variable: regresión de la anomalía por temporada sobre el ONI
(sin tendencia en el ONI, para no remover tendencia por colinealidad), y tendencia de Sen
+ MK modificado sobre el residuo. Se reporta también la sensibilidad a ENSO (°C por °C de ONI).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen  # noqa: E402


def oni_temporada():
    o = pd.read_csv("data/indices/oni.txt", sep=r"\s+")
    o = o[o.SEAS.isin(["OND", "NDJ", "DJF", "JFM"])].copy()
    o["temporada"] = np.where(o.SEAS.isin(["OND", "NDJ"]), o.YR + 1, o.YR)
    s = o.groupby("temporada").ANOM.mean()
    n = o.groupby("temporada").size()
    return s[n == 4]


def main():
    oni = oni_temporada()
    est = pd.read_csv("analisis/03_tendencias_anomalias_aj_pares.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    series = pd.read_csv("analisis/03_series_temporada_anomalias_aj_pares.csv")
    series = series[series.temp_tipo == "ONDJFM"]
    filas = []
    for _, e in est.iterrows():
        s = series[series.sid == e.sid].set_index("temporada")
        fila = {"nombre": e.nombre, "lon": e.lon}
        for v in ("t", "td", "tw", "frac_humedas"):
            y = s[v]
            x = oni.reindex(y.index)
            ok = y.notna() & x.notna()
            yy, xx = y[ok].values, x[ok].values
            anios = y[ok].index.values.astype(float)
            # ONI sin tendencia (por si tuviera) antes de regresar
            xx_d = xx - np.polyval(np.polyfit(anios, xx, 1), anios)
            b = np.polyfit(xx_d, yy - yy.mean(), 1)[0]
            resid = yy - b * xx_d
            fila[f"{v}_enso_coef"] = b
            fila[f"{v}_sen_bruta"] = sen(anios, yy) * 10
            fila[f"{v}_sen_sin_enso"] = sen(anios, resid) * 10
            fila[f"{v}_p_sin_enso"] = mk_hamed_rao(anios, resid)[1]
            fila[f"{v}_r_enso"] = np.corrcoef(xx_d, yy)[0, 1]
        filas.append(fila)
    res = pd.DataFrame(filas).sort_values("lon")
    res.to_csv("analisis/11_enso.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["nombre", "t_r_enso", "td_r_enso", "tw_r_enso", "td_sen_bruta", "td_sen_sin_enso",
            "tw_sen_bruta", "tw_sen_sin_enso", "frac_humedas_sen_bruta", "frac_humedas_sen_sin_enso"]
    print(res[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
