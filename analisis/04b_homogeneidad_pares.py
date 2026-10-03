"""Homogeneidad por comparación de pares (versión simplificada de PHA, Menne & Williams 2009).

Para cada candidata y variable (T y DPD):
1. Vecinas: hasta 7, a <500 km, r > 0.5 en primeras diferencias de anomalías mensuales.
2. Para cada par, serie diferencia candidata - vecina y segmentación SNHT recursiva con
   valor crítico AR(1) (misma función que 04_homogeneidad.py).
3. Atribución: un quiebre se asigna a la candidata si aparece (±12 meses) en al menos la
   mitad de sus pares y en no menos de 2. Así un quiebre de una vecina (que aparece en un
   solo par) no se le atribuye a la candidata.
4. Magnitud: mediana de los saltos estimados en cada par que lo confirma.
"""
import glob
import os
from importlib import import_module

import numpy as np
import pandas as pd

hom = import_module("04_homogeneidad")
VENTANA = pd.Timedelta(days=365)


def quiebres_par(dif):
    out = []
    hom.segmentar(dif.values, dif.index, out)
    return [(q[0], q[1]) for q in out]


def agrupar(eventos, n_pares):
    """eventos: lista de (fecha, salto, id_par). Agrupa fechas cercanas y exige confirmación."""
    eventos = sorted(eventos)
    grupos, actual = [], []
    for e in eventos:
        if actual and e[0] - actual[0][0] > VENTANA * 2:
            grupos.append(actual)
            actual = []
        actual.append(e)
    if actual:
        grupos.append(actual)
    aceptados = []
    minimo = max(2, int(np.ceil(n_pares / 2)))
    for g in grupos:
        pares = {e[2] for e in g}
        if len(pares) >= minimo:
            fechas = pd.to_datetime([e[0] for e in g])
            fecha = fechas.sort_values()[len(fechas) // 2]
            salto = float(np.median([e[1] for e in g]))
            aceptados.append((fecha, salto, len(pares)))
    return aceptados


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    tend = pd.read_csv(os.environ.get("CANDIDATAS", "analisis/03_tendencias_anomalias.csv"))
    candidatas = tend[(tend.temporada == "ONDJFM") & tend.incluida].sid.tolist()
    todas = [os.path.basename(f)[:-8] for f in glob.glob("data/noches/*.parquet")]
    series = {s: hom.mensual(s) for s in todas}
    filas = []
    for c in candidatas:
        pc = (info.loc[c, "lat"], info.loc[c, "lon"])
        for v in ("t", "dpd", "td"):
            xc = series[c][v]
            vec = []
            for s in todas:
                if s == c:
                    continue
                if hom.distancia_km(pc, (info.loc[s, "lat"], info.loc[s, "lon"])) > 500:
                    continue
                xs = series[s][v].reindex(xc.index)
                par = pd.concat([xc.diff(), xs.diff()], axis=1).dropna()
                if len(par) < 120:
                    continue
                r = par.corr().iloc[0, 1]
                if r > 0.5:
                    vec.append((r, s))
            vec = sorted(vec, reverse=True)[:7]
            eventos = []
            for _, s in vec:
                dif = (xc - series[s][v].reindex(xc.index)).dropna()
                if len(dif) < 120:
                    continue
                for fecha, salto in quiebres_par(dif):
                    eventos.append((fecha, salto, s))
            acept = agrupar(eventos, len(vec)) if len(vec) >= 2 else []
            filas.append({"sid": c, "nombre": info.loc[c, "nombre"], "var": v, "n_pares": len(vec),
                          "vecinas_sid": ";".join(s for _, s in vec),
                          "quiebres": "; ".join(f"{f:%Y-%m} ({x:+.2f} °C)" for f, x, _ in acept),
                          "confirmaciones": ";".join(str(n) for _, _, n in acept),
                          "n_quiebres": len(acept)})
            print(c, info.loc[c, "nombre"].strip(), v, filas[-1]["quiebres"], flush=True)
    pd.DataFrame(filas).to_csv(os.environ.get("SALIDA_HOM", "analisis/04b_homogeneidad_pares.csv"), index=False)


if __name__ == "__main__":
    main()
