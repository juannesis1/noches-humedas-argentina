"""Sensibilidad de la descomposición con tipos observados (12b) a la definición de los tipos.

Variantes (oct-dic, 1980-2002 vs 2003-2025): umbral atlántico 2 o 5 m/s; jet por percentil 75 (Montini et
al. 2019) en lugar de Bonner; tipos con el sondeo de Córdoba en lugar de Resistencia; dos tipos (norte vs
resto). Para cada una: cambio total, término de frecuencia y término dentro del tipo en las estaciones de
alta expansión (> 20 pp), y ρ de cada término con la expansión.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
d12 = importlib.import_module("12_descomposicion")
tend = importlib.import_module("03_tendencias")


def tipos(sid, umbral_atl=3.0, jet="bonner", dos=False):
    a = pd.read_parquet(f"data/igra/proc/{sid}.parquet")
    a = a[a.index.hour == 12].dropna(subset=["u850", "v850"])
    w8, w7 = np.hypot(a.u850, a.v850), np.hypot(a.u700, a.v700)
    cz = w8 - w7
    d = (np.degrees(np.arctan2(-a.u850, -a.v850)) + 360) % 360
    norte = (d >= 292.5) | (d <= 45)
    if jet == "bonner":
        es_jet = (w8 >= 12) & (cz >= 6) & norte
    else:
        base = a[(a.index.year >= 1981) & (a.index.year <= 2010)]
        bw = np.hypot(base.u850, base.v850)
        pw = bw.groupby(base.index.month).quantile(0.75)
        pc = (bw - np.hypot(base.u700, base.v700)).groupby(base.index.month).quantile(0.75)
        es_jet = (w8.values >= pw.reindex(a.index.month).values) & (cz.values >= pc.reindex(a.index.month).values) & norte.values
    if dos:
        t = np.where((d >= 292.5) | (d < 67.5), "NORTE", "RESTO")
    else:
        t = np.where(es_jet, "JET", np.where((d >= 22.5) & (d < 135) & (w8 >= umbral_atl), "ATLANTICO",
                     np.where((d >= 135) & (d < 292.5), "SUR", "NORTE")))
    s = pd.Series(t, index=a.index.normalize())
    return s[~s.index.duplicated()]


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    variantes = {"base (Resistencia, Bonner, 3 m/s)": dict(sid="ARM00087155"),
                 "atlántico ≥ 2 m/s": dict(sid="ARM00087155", umbral_atl=2.0),
                 "atlántico ≥ 5 m/s": dict(sid="ARM00087155", umbral_atl=5.0),
                 "jet P75 (Montini)": dict(sid="ARM00087155", jet="p75"),
                 "sondeo de Córdoba": dict(sid="ARM00087344"),
                 "dos tipos (norte / resto)": dict(sid="ARM00087155", dos=True)}
    p1, p2 = (1980, 2002), (2003, 2025)
    datos = {}
    for _, e in est.iterrows():
        d = pd.read_parquet(f"data/noches_ajustadas_pares_laxo/{e.sid}.parquet")
        datos[e.nombre.strip()] = tend.anomalias(d[d.index.month.isin([10, 11, 12])])
    filas = []
    for nombre, kw in variantes.items():
        tp = tipos(**kw)
        d12.TIPOS = sorted(set(tp.values))
        for n, d in datos.items():
            dd = d.copy()
            dd["tipo"] = tp.reindex(dd.index).values
            dd = dd.dropna(subset=["tipo"])
            filas.append({"variante": nombre, "nombre": n, **d12.descomponer(dd, "td", p1, p2)})
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left").dropna(subset=["delta_soja_pp"])
    r.to_csv("analisis/12c_descomposicion_sensibilidad.csv", index=False)
    for v, g in r.groupby("variante", sort=False):
        a = g[g.delta_soja_pp > 20]
        print(f"{v:32s} alta exp: total {a.total.mean():+.2f} = frec {a.frecuencia.mean():+.2f} + dentro {a.dentro.mean():+.2f} "
              f"| ρ dentro {stats.spearmanr(g.delta_soja_pp, g.dentro)[0]:+.2f}  ρ frec {stats.spearmanr(g.delta_soja_pp, g.frecuencia)[0]:+.2f}")


if __name__ == "__main__":
    main()
