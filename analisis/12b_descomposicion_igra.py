"""Descomposición frecuencia / dentro del tipo con tipos OBSERVADOS (radiosondeo de Resistencia).

Motivo (revisión, 2 de octubre): la detección del jet en ERA5 no es homogénea en el tiempo (en Resistencia
POD 0.22 en 1979-89, 0.30 en 2002-09 y 0.69 en 2016-25; FAR 0.47 → 0.09), así que la frecuencia de tipos
de ERA5 puede tener tendencias espurias. Aquí cada noche se tipifica con el sondeo de las 12 UTC de esa
mañana en Resistencia (27.45°S, 59.05°W, centro de la región):
  JET       : criterio de Bonner (|V|850 ≥ 12, dirección 292.5-45°, |V|850 − |V|700 ≥ 6)
  ATLANTICO : viento a 850 hPa del NE al SE (22.5-135°) y ≥ 3 m/s
  NORTE     : resto con componente norte (292.5-22.5°)
  SUR       : 135-292.5° (post-frontal)
Mismo cálculo que 12_descomposicion.py (anomalías diarias, bootstrap por temporadas). Los sondeos cubren
1980-2025, así que se usan los períodos definitivos 1980-2002 vs 2003-2025 (y 1980-89 vs 2017-25 para
comparar con la versión ERA5). Limitación: un solo punto; el flujo a 850 hPa en Resistencia es
representativo del transporte hacia la región, pero no de toda ella.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
d12 = importlib.import_module("12_descomposicion")
tend = importlib.import_module("03_tendencias")


def tipos_igra():
    a = pd.read_parquet("data/igra/proc/ARM00087155.parquet")
    a = a[a.index.hour == 12].dropna(subset=["u850", "v850"])
    w8 = np.hypot(a.u850, a.v850)
    w7 = np.hypot(a.u700, a.v700)
    d = (np.degrees(np.arctan2(-a.u850, -a.v850)) + 360) % 360
    jet = (w8 >= 12) & ((w8 - w7) >= 6) & ((d >= 292.5) | (d <= 45))
    tipo = np.where(jet, "JET", np.where((d >= 22.5) & (d < 135) & (w8 >= 3), "ATLANTICO",
                    np.where((d >= 135) & (d < 292.5), "SUR", "NORTE")))
    t = pd.Series(tipo, index=a.index.normalize())
    return t[~t.index.duplicated()]


def main():
    tipo = tipos_igra()
    temp = np.where(tipo.index.month >= 10, tipo.index.year + 1, tipo.index.year)
    m = np.isin(tipo.index.month, [10, 11, 12, 1, 2, 3])
    print("Frecuencia de tipos (sondeo Resistencia, oct-mar) por década:")
    print(pd.crosstab((temp[m] // 10) * 10, tipo.values[m], normalize="index").round(3))
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    filas = []
    for (p1, p2) in (((1980, 2002), (2003, 2025)), ((1980, 1989), (2017, 2025))):
        for vn, meses in (("OND", [10, 11, 12]), ("ONDJFM", [10, 11, 12, 1, 2, 3])):
            for _, e in est.iterrows():
                d = pd.read_parquet(f"data/noches_ajustadas_pares_laxo/{e.sid}.parquet")
                d = d[d.index.month.isin(meses)]
                d = tend.anomalias(d)
                d["tipo"] = tipo.reindex(d.index).values
                d = d.dropna(subset=["tipo"])
                r = d12.descomponer(d, "td", p1, p2)
                b = d12.bootstrap(d, "td", p1, p2, n=200)
                filas.append({"periodos": f"{p1[0]}-{p1[1]} vs {p2[0]}-{p2[1]}", "ventana": vn,
                              "nombre": e.nombre.strip(), **r,
                              "dentro_lo": b.dentro.quantile(0.025), "dentro_hi": b.dentro.quantile(0.975)})
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    r.to_csv("analisis/12b_descomposicion_igra.csv", index=False)
    from scipy import stats
    for (pp, vn), g in r.groupby(["periodos", "ventana"], sort=False):
        q = g.dropna(subset=["delta_soja_pp"])
        alta, baja = q[q.delta_soja_pp > 20], q[q.delta_soja_pp < 5]
        print(f"\n{pp} {vn}: alta exp total {alta.total.mean():+.2f} = frec {alta.frecuencia.mean():+.2f} + dentro "
              f"{alta.dentro.mean():+.2f} | baja total {baja.total.mean():+.2f} = {baja.frecuencia.mean():+.2f} + "
              f"{baja.dentro.mean():+.2f} | ρ(cultivos, dentro) {stats.spearmanr(q.delta_soja_pp, q.dentro)[0]:+.2f} "
              f"ρ(cultivos, frec) {stats.spearmanr(q.delta_soja_pp, q.frecuencia)[0]:+.2f} | dentro con IC95<0: "
              f"{(q.dentro_hi < 0).sum()} est.")


if __name__ == "__main__":
    main()
