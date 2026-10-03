"""¿Cambiaron también las noches húmedas extremas, o solo el promedio?

Para cada estación (Td nocturna homogeneizada, oct-dic, anomalías respecto de la climatología diaria 1981-2010):
tendencias de los percentiles 10, 50 y 90 de cada temporada (Sen), y de la frecuencia de noches con Tw ≥ P90
propio (1981-2010) en oct-dic. Relación con la expansión de cultivos. Para salud importan las colas altas.
"""
import os
import sys
import importlib

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

tend = importlib.import_module("03_tendencias")


def tendencia(s):
    s = s[(s.index >= 1980) & (s.index <= 2025)].dropna()
    ok = len(s) >= 30 and (s.index <= 1989).sum() >= 5 and (s.index >= 2016).sum() >= 5
    return sen(s.index.values.astype(float), s.values) * 10 if ok else np.nan


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    filas = []
    for _, e in est.iterrows():
        d = pd.read_parquet(f"data/noches_ajustadas_pares_laxo/{e.sid}.parquet")
        d = d[d.index.month.isin([10, 11, 12])]
        base = d[(d.temporada >= 1981) & (d.temporada <= 2010)]
        p90tw = base.tw.quantile(0.9)
        a = tend.anomalias(d)
        g = a.groupby("temporada")
        n = g.size()
        ok = n[n >= 46].index
        fila = {"nombre": e.nombre.strip(), "lat": e.lat, "lon": e.lon}
        for q in (0.1, 0.5, 0.9):
            fila[f"td_p{int(q * 100)}"] = tendencia(g.td.quantile(q).loc[ok])
        fila["humedas_pp"] = tendencia((d.assign(h=d.tw >= p90tw).groupby("temporada").h.mean() * 100).loc[ok])
        filas.append(fila)
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    r.to_csv("analisis/31_extremos.csv", index=False)
    q = r.dropna(subset=["delta_soja_pp"])
    for c in ("td_p10", "td_p50", "td_p90", "humedas_pp"):
        a, b = q[q.delta_soja_pp > 20][c].mean(), q[q.delta_soja_pp < 5][c].mean()
        print(f"{c:11s} alta exp {a:+.2f}  baja {b:+.2f}  ρ con cultivos {stats.spearmanr(q.delta_soja_pp, q[c], nan_policy='omit')[0]:+.2f}")


if __name__ == "__main__":
    main()
