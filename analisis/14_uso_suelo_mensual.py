"""Prueba mecanística: ¿la relación secado–expansión de cultivos de verano depende del mes?

Si el mecanismo es el reemplazo de pasturas perennes por cultivos anuales de verano, el
efecto debería ser mayor cuando los lotes están desnudos o con poca cobertura (oct-dic,
antes del cierre del canopeo; Pal et al. 2021 encuentran las mayores diferencias de flujos
en nov-dic) que en ene-mar.
Para cada estación y mes: tendencia de Sen de la anomalía mensual de Td nocturna
(1980-2025, series homogeneizadas). Luego correlación (Spearman y parcial lat/lon) con el
cambio de fracción de cultivos de verano en 100 km (13_uso_suelo.py, CULTIVO=verano).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402
from importlib import import_module  # noqa: E402

tend = import_module("03_tendencias")


def main():
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv")
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & est.sid.str.startswith("87")]
    est = est[~est.nombre.str.contains("CERES")].set_index("nombre")
    filas = []
    for nombre, e in est.iterrows():
        d = pd.read_parquet(f"data/noches_ajustadas_pares_laxo/{e.sid}.parquet")
        d = tend.anomalias(d[d.index.month.isin([10, 11, 12, 1, 2, 3])])
        m = d.groupby([d.temporada, d.index.month]).td.agg(["mean", "size"]).reset_index()
        m.columns = ["temporada", "mes", "td", "n"]
        m = m[(m.n >= 10) & m.temporada.between(1980, 2025)]
        fila = {"nombre": nombre}
        for mes in (10, 11, 12, 1, 2, 3):
            s = m[m.mes == mes]
            fila[mes] = sen(s.temporada.values.astype(float), s.td.values) * 10 if len(s) > 30 else np.nan
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("nombre")
    u = uso.set_index("nombre")
    t = t.join(u[["lat", "lon", "delta_soja_pp"]])
    print("Tendencia de Td nocturna por mes (°C/década):")
    print(t[[10, 11, 12, 1, 2, 3, "delta_soja_pp"]].round(2).sort_values("delta_soja_pp").to_string())
    X = np.c_[np.ones(len(t)), t.lat, t.lon]
    rx = t.delta_soja_pp - X @ np.linalg.lstsq(X, t.delta_soja_pp, rcond=None)[0]
    res = []
    for mes in (10, 11, 12, 1, 2, 3):
        ok = t[mes].notna()
        r, p = stats.spearmanr(t.delta_soja_pp[ok], t[mes][ok])
        ry = t[mes][ok] - X[ok.values] @ np.linalg.lstsq(X[ok.values], t[mes][ok], rcond=None)[0]
        rp, pp = stats.pearsonr(rx[ok], ry)
        res.append({"mes": mes, "rho": r, "p": p, "r_parcial": rp, "p_parcial": pp,
                    "td_media_alta_exp": t[mes][t.delta_soja_pp > 15].mean(),
                    "td_media_baja_exp": t[mes][t.delta_soja_pp < 5].mean()})
    res = pd.DataFrame(res)
    res.to_csv("analisis/14_uso_suelo_mensual.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
