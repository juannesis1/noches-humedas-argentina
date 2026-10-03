"""Ajuste de inhomogeneidades y reconstrucción de las noches ajustadas.

Siguiendo la lógica de HadISDH (Willett et al. 2014): los quiebres se detectan en T y en
DPD (mejor relación señal/ruido) y se aplican de forma consistente a T y a Td.
Para cada quiebre, el ajuste de cada variable es la diferencia de medias de la serie
diferencia (candidata - mediana de vecinas) entre los 5 años posteriores y los 5 años
anteriores al quiebre (o lo disponible, mínimo 24 meses de cada lado). Se ajusta hacia
atrás, tomando como referencia el segmento más reciente. Td se ajusta como T - DPD.
Luego se recalculan Tw, q y HR con los valores diarios ajustados.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import termo  # noqa: E402
from importlib import import_module  # noqa: E402

hom = import_module("04_homogeneidad")


def fusionar(fechas, meses=24):
    """Quiebres a menos de 24 meses entre sí se tratan como un único evento (el primero)."""
    out = []
    for q in sorted(fechas):
        if not out or (q - out[-1]).days > meses * 30.4:
            out.append(q)
    return out


def ajustes_variable(dif, fechas_q):
    """Salto (posterior - anterior) en la serie diferencia para cada quiebre, usando solo
    el segmento propio a cada lado (sin cruzar quiebres vecinos) y como máximo 5 años."""
    out = {}
    fq = sorted(fechas_q)
    for i, q in enumerate(fq):
        ini = max(fq[i - 1] if i > 0 else pd.Timestamp("1900-01-01"), q - pd.DateOffset(years=5))
        fin = min(fq[i + 1] if i + 1 < len(fq) else pd.Timestamp("2100-01-01"),
                  q + pd.DateOffset(years=5))
        antes = dif[(dif.index < q) & (dif.index >= ini)].dropna()
        despues = dif[(dif.index >= q) & (dif.index < fin)].dropna()
        if len(antes) >= 12 and len(despues) >= 12:
            out[q] = despues.mean() - antes.mean()
    return out


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    metodo = os.environ.get("HOM", "pares")
    fuente = os.environ.get("FUENTE_HOM") or ("analisis/04b_homogeneidad_pares.csv" if metodo == "pares" else "analisis/04_homogeneidad.csv")
    h = pd.read_csv(fuente)
    salida = os.environ.get("SALIDA_AJ", f"data/noches_ajustadas_{metodo}")
    os.makedirs(salida, exist_ok=True)
    base = os.environ.get("DIR_BASE", "data/noches")   # data/dias: mismos quiebres, ajuste diurno propio
    todas = [f[:-8] for f in os.listdir(base) if f.endswith(".parquet")]
    series = {s: hom.mensual(s) for s in todas}
    registro = []
    for sid in h.sid.unique():
        hs = h[h.sid == sid].set_index("var")
        fechas = set()
        for v in ("t", "dpd"):
            q = hs.loc[v, "quiebres"] if v in hs.index else np.nan
            if isinstance(q, str) and q.strip():
                fechas |= {pd.Timestamp(x.split(" ")[0]) for x in q.split("; ")}
        d = pd.read_parquet(f"{base}/{sid}.parquet")
        fechas = fusionar(fechas)
        if fechas:
            ajuste = {"t": {}, "dpd": {}}
            for v in ("t", "dpd"):
                vs = hs.loc[v, "vecinas_sid"] if v in hs.index else np.nan
                if not isinstance(vs, str) or not vs:
                    continue
                ids_vec = vs.split(";")
                if len(ids_vec) < 2:
                    continue
                ref = pd.concat([series[s][v] for s in ids_vec], axis=1).median(axis=1)
                dif = series[sid][v] - ref.reindex(series[sid].index)
                ajuste[v] = ajustes_variable(dif, fechas)
            for q in sorted(fechas):
                at, ad = ajuste["t"].get(q, 0.0), ajuste["dpd"].get(q, 0.0)
                antes = d.index < q
                d.loc[antes, "t"] += at
                d.loc[antes, "dpd"] += ad
                registro.append({"sid": sid, "nombre": info.loc[sid, "nombre"], "fecha": q.date(),
                                 "ajuste_T": round(at, 2), "ajuste_DPD": round(ad, 2)})
            d["td"] = d.t - d.dpd
            p = termo.presion_estandar(info.loc[sid, "elev"])
            d["tw"] = termo.tw_davies_jones(d.t.values, d.td.values, p)
            d["q"] = termo.q_desde_td(d.td.values, p)
            d["hr"] = termo.rh_desde_td(d.t.values, d.td.values)
        d.to_parquet(f"{salida}/{sid}.parquet")
    pd.DataFrame(registro).to_csv(f"analisis/05_ajustes_{metodo}{os.environ.get('SUFIJO', '')}.csv", index=False)
    print(pd.DataFrame(registro).to_string(index=False))


if __name__ == "__main__":
    main()
