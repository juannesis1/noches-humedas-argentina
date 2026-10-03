"""¿Explica la variabilidad decadal del Pacífico (PDO) el secado de oct-dic?

Disparador (revisión, 2 de octubre): la humedad a 850 hPa fue alta en los 80 y baja en los 2000, en
coincidencia con el cambio de fase de la PDO (positiva hasta fines de los 90, negativa después), que
modula la frecuencia del jet central (Mu et al. 2024). Si el secado fuera decadal del Pacífico:
(a) la Td de oct-dic correlacionaría con la PDO en todas las estaciones;
(b) remover ENSO + SAM + PDO (con su tendencia, que es negativa en 1980-2025) eliminaría el secado y su
    relación con la agricultura. Los coeficientes se estiman sobre series sin tendencia (variabilidad
    interanual) y luego se resta el índice completo, igual que para el SAM en 11b_enso_sam.py.
Índice: ERSST v5 PDO (NOAA NCEI), media oct-dic de cada año (temporada = año + 1, como en el resto).
ONI y SAM como en 11b_enso_sam.py, en sus medias oct-dic. Tendencias Sen; Td canónica homogeneizada.
También se correlaciona la PDO con la humedad a 850 hPa de los radiosondeos (12 UTC, oct-dic).
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen  # noqa: E402
import termo  # noqa: E402

h = importlib.import_module("19_huella_temporal")
nul = importlib.import_module("24_nulo_espacial")


def mensual_a_ond(tabla):
    """tabla: índice año, columnas 1..12 → media oct-dic del año a, asignada a la temporada a+1."""
    v = tabla[[10, 11, 12]].mean(axis=1)
    v.index = v.index + 1
    return v


def pdo():
    t = pd.read_csv("data/indices/pdo_ersst.dat", sep=r"\s+", skiprows=1)
    t.columns = ["anio"] + list(range(1, 13))
    t = t.set_index("anio").replace(99.99, np.nan)
    return mensual_a_ond(t)


def oni():
    t = pd.read_csv("data/indices/oni.txt", sep=r"\s+")
    # formato CPC: SEAS YR TOTAL ANOM; OND está centrado en noviembre
    t = t[t.SEAS == "OND"].set_index("YR").ANOM
    t.index = t.index + 1
    return t


def sam():
    s = pd.read_csv("data/indices/sam_marshall.txt", sep=r"\s+", skiprows=2, header=None,
                    names=["anio"] + list(range(1, 13))).set_index("anio")
    return mensual_a_ond(s)


def main():
    P, O, S = pdo(), oni(), sam()
    idx = pd.concat([P.rename("pdo"), O.rename("oni"), S.rename("sam")], axis=1).loc[1980:2025]
    a = idx.index.values.astype(float)
    print("Tendencias de índices oct-dic 1980-2025 (por década):",
          {c: round(sen(a, idx[c].values) * 10, 2) for c in idx})
    print("Correlación PDO–ONI oct-dic:", round(idx.pdo.corr(idx.oni), 2))
    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    h.DIR = "data/noches_ajustadas_pares_laxo"
    filas = []
    for _, e in est.iterrows():
        y = h.serie(e.sid, "td", [10, 11, 12])
        df = pd.concat([y.rename("td"), idx], axis=1).dropna()
        x = df.index.values.astype(float)
        # coeficientes estimados con índices Y Td SIN tendencia (solo variabilidad interanual), para que la
        # tendencia compartida no se atribuya a los modos por colinealidad; luego se resta el modo completo
        st = lambda v: v - np.polyval(np.polyfit(x, v, 1), x)
        Xd = np.c_[np.ones(len(df)), st(df.oni.values), st(df.sam.values), st(df.pdo.values)]
        b = np.linalg.lstsq(Xd, st(df.td.values), rcond=None)[0]
        X = np.c_[np.ones(len(df)), df.oni, df.sam, df.pdo]
        resid = df.td.values - X[:, 1:] @ b[1:]
        fila = {"nombre": e.nombre.strip(), "lat": e.lat, "lon": e.lon,
                "r_pdo": stats.pearsonr(df.td, df.pdo)[0],
                "b_pdo": b[3], "td_ond": sen(x, df.td.values) * 10,
                "td_ond_sin_modos": sen(x, resid) * 10,
                "aporte_pdo": b[3] * sen(x, df.pdo.values) * 10,
                "aporte_sam": b[2] * sen(x, df.sam.values) * 10}
        filas.append(fila)
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    r.to_csv("analisis/28_pdo.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.sort_values("lat").round(3).to_string(index=False))
    q = r.dropna(subset=["delta_soja_pp"])
    D = nul.dist(q.lat.values, q.lon.values)
    for c in ("td_ond", "td_ond_sin_modos", "aporte_pdo"):
        print(f"Δsembrado vs {c:17s}: ρ {stats.spearmanr(q.delta_soja_pp, q[c])[0]:+.2f}")
    alta = q[q.delta_soja_pp > 20]
    print(f"Alta expansión: tendencia {alta.td_ond.mean():+.3f}, sin ENSO/SAM/PDO {alta.td_ond_sin_modos.mean():+.3f}, "
          f"aporte PDO {alta.aporte_pdo.mean():+.3f}, aporte SAM {alta.aporte_sam.mean():+.3f} °C/déc")
    print("\nRadiosondeos q850 oct-dic (12 UTC) vs PDO:")
    for sid, n in (("ARM00087344", "Córdoba"), ("ARM00087155", "Resistencia"), ("ARM00087576", "Ezeiza"),
                   ("ARM00087623", "Santa Rosa")):
        d = pd.read_parquet(f"data/igra/proc/{sid}.parquet")
        d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12])].dropna(subset=["t850", "dpd850"])
        q850 = pd.Series(termo.q_desde_td((d.t850 - d.dpd850).values, 850.0), index=d.index)
        g = q850.groupby(q850.index.year + 1)
        s = g.mean()[g.count() >= 25].loc[1980:2025]
        j = pd.concat([s.rename("q"), idx.pdo], axis=1).dropna()
        print(f"  {n:12s} r(q850, PDO) = {j.q.corr(j.pdo):+.2f} (n={len(j)})")


if __name__ == "__main__":
    main()
