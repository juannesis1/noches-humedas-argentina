"""Serie anual de la diferencia entre estaciones de alta (> 20 pp) y baja (< 5 pp) expansión de cultivos de verano.

Antes este CSV (Fig. 5) se había generado a mano; la revisión I-2/I-3 (3 de octubre) pide un script y un umbral único
(> 20 / < 5 pp, el mismo de la descomposición, el diagnóstico diurno y el dipolo).
Para cada temporada: media de las anomalías de Td nocturna (homogeneizada) de oct-dic y ene-mar en cada grupo, y la
diferencia de fracción de cultivos de verano en 100 km. Tendencias de Sen de la diferencia oct-dic por subperíodo.
Salidas: analisis/19_diferencia_grupos.csv y analisis/19b_resumen.txt. UMBRALES=alta,baja permite sensibilidad.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

h = importlib.import_module("19_huella_temporal")
SENS = bool(os.environ.get("UMBRALES"))
ALTA, BAJA = (float(x) for x in (os.environ.get("UMBRALES") or "20,5").split(","))


def main():
    h.DIR = "data/noches_ajustadas_pares_laxo"
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")].copy()
    est["nombre"] = est.nombre.str.strip()
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    est = est.merge(uso[["nombre", "delta_soja_pp"]], on="nombre").dropna(subset=["delta_soja_pp"])
    fr = pd.read_csv("analisis/19_fraccion_anual_verano.csv", index_col="anio")
    fr.columns = fr.columns.str.strip()
    grupos = {"alta": est[est.delta_soja_pp > ALTA], "baja": est[est.delta_soja_pp < BAJA]}
    out = {}
    for g, e in grupos.items():
        for vn, meses in (("OND", [10, 11, 12]), ("JFM", [1, 2, 3])):
            out[(g, vn)] = pd.concat({n: h.serie(s, "td", meses) for s, n in zip(e.sid, e.nombre)}, axis=1).mean(axis=1)
        out[(g, "cult")] = fr[[n for n in e.nombre if n in fr.columns]].mean(axis=1)
    d = pd.DataFrame({"OND": out[("alta", "OND")] - out[("baja", "OND")],
                      "JFM": out[("alta", "JFM")] - out[("baja", "JFM")]})
    d["dcult"] = (out[("alta", "cult")] - out[("baja", "cult")]).reindex(d.index)
    d.index.name = "temporada"
    d = d[(d.index >= 1980) & (d.index <= 2025)]
    if not SENS:
        d.to_csv("analisis/19_diferencia_grupos.csv")
    L = [f"Grupos: alta > {ALTA:g} pp (n {len(grupos['alta'])}), baja < {BAJA:g} pp (n {len(grupos['baja'])})"]
    for a, b in ((1980, 1995), (1996, 2010), (2011, 2025), (1980, 2025)):
        s = d.OND.loc[a:b].dropna()
        c = d.dcult.loc[a:b].dropna()
        L.append(f"  {a}-{b}: Td oct-dic alta−baja {sen(s.index.values.astype(float), s.values) * 10:+.2f} °C/déc; "
                 f"cultivos {sen(c.index.values.astype(float), c.values) * 10:+.1f} pp/déc")
    m7 = d.OND.rolling(7, center=True, min_periods=7).mean()
    L.append(f"  Media 7 años oct-dic: máximo {m7.max():+.2f} en {m7.idxmax()}, 1995 {m7.get(1995, np.nan):+.2f}, "
             f"2010 {m7.get(2010, np.nan):+.2f}, último {m7.dropna().iloc[-1]:+.2f}")
    txt = "\n".join(L)
    if not SENS:
        open("analisis/19b_resumen.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
