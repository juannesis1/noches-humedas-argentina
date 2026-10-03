"""Relación entre el cambio de cobertura (MapBiomas) y las tendencias nocturnas de Td.

Responde a M8 de la revisión estricta: medir directamente el mecanismo propuesto (pérdida de
coberturas herbáceas perennes) y no solo la superficie sembrada.
Métricas por estación (radio 50/100/150 km), cambio 1985-89 → 2015-19 en puntos porcentuales:
  cultivos      = c19 (cultivos temporarios) + c36 (perennes) + c18
  perennes_herb = c12 (herbáceas) + c11 (herbáceas inundables) + c15 (pastura)
  pastura       = c15
  mosaico       = c21
  bosque        = c3 + c4 + c6
  urbano        = c24
Correlaciones (Spearman y parcial lat/lon) con las tendencias de Td oct-dic, oct-mar y ene-mar
(series canónicas homogeneizadas, Sen 1980-2025), significancia con campos sustitutos
(24_nulo_espacial). Estaciones con < 40 % del círculo dentro de Argentina se excluyen.
Comparación con la superficie sembrada (MAGyP) para validar la métrica.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

h = importlib.import_module("19_huella_temporal")
nul = importlib.import_module("24_nulo_espacial")
GRUPOS = {"cultivos": ["c19", "c36", "c18"], "perennes_herb": ["c12", "c11", "c15"], "pastura": ["c15"],
          "mosaico": ["c21"], "bosque": ["c3", "c4", "c6"], "urbano": ["c24"]}
P1, P2 = (1985, 1989), (2015, 2019)
rng = np.random.default_rng(7)


def parcial(x, y, Z):
    X = np.c_[np.ones(len(x)), Z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0]


def p_sustitutos(x, y, D, n=3000):
    xr = stats.rankdata(x).astype(float)
    a = nul.alcance(xr, D)
    L = np.linalg.cholesky(np.exp(-D / a) + 1e-9 * np.eye(len(x)))
    sims = L @ rng.standard_normal((len(x), n))
    rho = stats.spearmanr(x, y)[0]
    rs = np.array([stats.spearmanr(sims[:, k], y)[0] for k in range(n)])
    return np.mean(np.abs(rs) >= abs(rho))


def main():
    c = pd.read_csv("data/mapbiomas/composicion.csv").fillna(0.0)
    for g, cols in GRUPOS.items():
        c[g] = c[[k for k in cols if k in c]].sum(axis=1) * 100
    anios = set(c.anio)
    assert all(y in anios for y in range(P1[0], P1[1] + 1)) and all(y in anios for y in range(P2[0], P2[1] + 1)), \
        "faltan años clave"
    m1 = c[c.anio.between(*P1)].groupby(["nombre", "radio"])[list(GRUPOS) + ["cobertura"]].mean()
    m2 = c[c.anio.between(*P2)].groupby(["nombre", "radio"])[list(GRUPOS)].mean()
    delta = (m2 - m1).add_prefix("d_").join(m1[["cobertura"]]).join(m1[list(GRUPOS)].add_prefix("ini_")).reset_index()

    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")].copy()
    est["nombre"] = est.nombre.str.strip()
    h.DIR = "data/noches_ajustadas_pares_laxo"
    for vn, meses in (("ond", [10, 11, 12]), ("ondjfm", [10, 11, 12, 1, 2, 3]), ("jfm", [1, 2, 3])):
        est[f"td_{vn}"] = [sen(s.index.values.astype(float), s.values) * 10
                           for s in (h.serie(sid, "td", meses) for sid in est.sid)]
    magyp = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    d = delta.merge(est[["nombre", "lat", "lon", "td_ond", "td_ondjfm", "td_jfm"]], on="nombre")
    d = d.merge(magyp[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    d = d[d.cobertura >= 0.4]
    d.to_csv("analisis/26_mapbiomas_delta.csv", index=False)

    pd.set_option("display.width", 250)
    q = d[d.radio == 100].sort_values("d_cultivos")
    print(q[["nombre", "cobertura", "ini_cultivos", "d_cultivos", "ini_perennes_herb", "d_perennes_herb",
             "d_pastura", "d_bosque", "delta_soja_pp", "td_ond"]].round(2).to_string(index=False))
    print(f"\nValidación: Δcultivos MapBiomas vs Δsembrada MAGyP (100 km): ρ = "
          f"{stats.spearmanr(q.d_cultivos, q.delta_soja_pp, nan_policy='omit')[0]:+.2f} (n={q.delta_soja_pp.notna().sum()})")
    filas = []
    for radio in (50, 100, 150):
        q = d[d.radio == radio]
        D = nul.dist(q.lat.values, q.lon.values)
        for g in GRUPOS:
            for vn in ("ond", "ondjfm", "jfm"):
                x, y = q[f"d_{g}"].values, q[f"td_{vn}"].values
                if np.std(x) == 0:
                    continue
                filas.append({"radio": radio, "metrica": g, "ventana": vn, "n": len(q),
                              "rho": stats.spearmanr(x, y)[0],
                              "parcial": parcial(x, y, np.c_[q.lat, q.lon]),
                              "p_sustitutos": p_sustitutos(x, y, D) if radio == 100 and vn != "jfm" else np.nan})
    r = pd.DataFrame(filas)
    r.to_csv("analisis/26_mapbiomas_correlaciones.csv", index=False)
    print("\nCorrelación cambio de cobertura vs tendencia de Td (radio 100 km):")
    print(r[r.radio == 100].pivot_table(index="metrica", columns="ventana", values=["rho", "parcial"]).round(2).to_string())
    print("\np sustitutos (100 km):")
    print(r[(r.radio == 100) & r.p_sustitutos.notna()][["metrica", "ventana", "rho", "p_sustitutos"]].round(3).to_string(index=False))
    print("\nSensibilidad al radio (ρ oct-dic):")
    print(r[r.ventana == "ond"].pivot(index="metrica", columns="radio", values="rho").round(2).to_string())


if __name__ == "__main__":
    main()
