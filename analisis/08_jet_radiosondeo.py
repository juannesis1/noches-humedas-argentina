"""Índice de jet observado con radiosondeos de 12 UTC (oct-mar, temporadas 1980-2025).

Criterios por sondeo:
- "bonner": |V|850 ≥ 12 m/s, dirección 850 entre 292.5° y 45° (NO a NE, Montini et al.
  2019), y |V|850 - |V|700 ≥ 6 m/s (Bonner 1968; Salio et al. 2002 a escala de estación).
- "p75": |V|850 y la cizalladura 850-700 por encima de su percentil 75 de la temporada
  (calculado por estación y por mes, 1981-2010), con la misma condición de dirección
  (Montini et al. 2019).
Frecuencia por temporada = % de sondeos válidos con jet; temporada válida con ≥60 sondeos.
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen  # noqa: E402

ESTACIONES = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087576": "Ezeiza",
              "BRM00083827": "Foz do Iguaçu", "BRM00083928": "Uruguaiana"}


def direccion(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


def main():
    filas, series = [], []
    for sid, nombre in ESTACIONES.items():
        a = pd.read_parquet(f"data/igra/proc/{sid}.parquet")
        a = a[(a.index.hour == 12)].dropna(subset=["u850", "v850", "u700", "v700"])
        w850 = np.hypot(a.u850, a.v850)
        w700 = np.hypot(a.u700, a.v700)
        dirn = direccion(a.u850, a.v850)
        norte = (dirn >= 292.5) | (dirn <= 45)
        cz = w850 - w700
        a["jet_bonner"] = (w850 >= 12) & (cz >= 6) & norte
        base = a[(a.index.year >= 1981) & (a.index.year <= 2010)]
        p75w = w850[base.index].groupby(base.index.month).quantile(0.75)
        p75c = cz[base.index].groupby(base.index.month).quantile(0.75)
        a["jet_p75"] = (w850.values >= p75w.reindex(a.index.month).values) & \
                       (cz.values >= p75c.reindex(a.index.month).values) & norte.values
        a["temporada"] = np.where(a.index.month >= 10, a.index.year + 1, a.index.year)
        g = a.groupby("temporada")
        f = pd.DataFrame({"n": g.size(), "bonner": g.jet_bonner.mean() * 100,
                          "p75": g.jet_p75.mean() * 100})
        f = f[(f.index >= 1980) & (f.index <= 2025) & (f.n >= 60)]
        f["estacion"] = nombre
        series.append(f.reset_index())
        x = f.index.values.astype(float)
        fila = {"estacion": nombre, "temporadas": len(f),
                "bonner_media_%": round(f.bonner.mean(), 1), "p75_media_%": round(f.p75.mean(), 1)}
        for c in ("bonner", "p75"):
            fila[f"{c}_tend_pp_dec"] = round(sen(x, f[c].values) * 10, 2)
            fila[f"{c}_p"] = round(mk_hamed_rao(x, f[c].values)[1], 3)
        filas.append(fila)
    res = pd.DataFrame(filas)
    pd.concat(series).to_csv("analisis/08_jet_radiosondeo_series.csv", index=False)
    res.to_csv("analisis/08_jet_radiosondeo.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
