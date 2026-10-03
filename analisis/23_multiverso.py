"""Curva de especificaciones (multiverso) para la relación expansión agrícola – secado nocturno.

Cada decisión analítica razonable se combina con todas las demás (Simonsohn et al. 2020,
"specification curve analysis"); se reporta la distribución completa de ρ y de la correlación
parcial (lat, lon), no una sola elección.
Decisiones:
  datos           crudos | homogeneizados por pares
  ventana         oct-dic | oct-mar
  estimador       Sen | mínimos cuadrados
  inicio          1980 | 1985
  radio           50 | 100 | 150 km
  cultivo         verano (soja+maíz+girasol) | soja | maíz
  períodos agro   1980-84→2015-19 | 1980-84→2010-14 | 1985-89→2015-19
  estaciones      todas (20) | sin las 5 de mayor crecimiento urbano (GHSL, 10 km)
Total: 2·2·2·2·3·3·3·2 = 864 especificaciones. Control: la misma curva con ene-mar (donde el
mecanismo predice ~0) y con trigo (placebo).
"""
import importlib
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

h = importlib.import_module("19_huella_temporal")

DATOS = {"crudos": "data/noches", "homog": "data/noches_ajustadas_pares_laxo"}
VENTANAS = {"OND": [10, 11, 12], "ONDJFM": [10, 11, 12, 1, 2, 3], "JFM": [1, 2, 3]}
CULTIVOS = {"verano": ["soja_1941_2024", "maiz", "girasol"], "soja": ["soja_1941_2024"],
            "maiz": ["maiz"], "trigo": ["trigo"]}
PERIODOS = {"80-84→15-19": ((1980, 1984), (2015, 2019)), "80-84→10-14": ((1980, 1984), (2010, 2014)),
            "85-89→15-19": ((1985, 1989), (2015, 2019))}


def parcial(x, y, Z):
    X = np.c_[np.ones(len(x)), Z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)[0]


def main():
    tend = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = tend[(tend.temporada == "ONDJFM") & tend.incluida & ~tend.nombre.str.contains("CERES")
               & tend.sid.str.startswith("87")][["sid", "nombre", "lat", "lon"]].reset_index(drop=True)
    urb = pd.read_csv("analisis/18_urbano.csv").set_index("nombre").db_10km
    est["db10"] = est.nombre.str.strip().map(urb)
    urbanas = set(est.sort_values("db10", ascending=False).nombre.head(5))
    print("Excluidas por urbanización:", sorted(urbanas))

    # series estacionales de Td por estación, datos y ventana (se calculan una vez)
    series = {}
    for dn, dd in DATOS.items():
        h.DIR = dd
        for vn, meses in VENTANAS.items():
            for _, e in est.iterrows():
                series[(dn, vn, e.nombre)] = h.serie(e.sid, "td", meses)

    # fracción anual de cada cultivo por radio
    delta = {}
    for radio in (50, 100, 150):
        h.RADIO_KM = radio
        W, sup = h.pesos(est)
        for cn, arch in CULTIVOS.items():
            f = h.fraccion_anual(est, W, sup, arch)
            f.index = f.index - 1                       # volver al año de campaña
            for pn, (p1, p2) in PERIODOS.items():
                delta[(radio, cn, pn)] = (f.loc[p2[0]:p2[1]].mean() - f.loc[p1[0]:p1[1]].mean())
        print("radio", radio, "listo", flush=True)

    filas = []
    for dn, vn, estim, ini, radio, cn, pn, conj in itertools.product(
            DATOS, VENTANAS, ("sen", "mco"), (1980, 1985), (50, 100, 150), CULTIVOS, PERIODOS,
            ("todas", "sin_urbanas")):
        sub = est if conj == "todas" else est[~est.nombre.isin(urbanas)]
        tds = []
        for n in sub.nombre:
            s = series[(dn, vn, n)]
            s = s[s.index >= ini]
            x = s.index.values.astype(float)
            tds.append((sen(x, s.values) if estim == "sen" else np.polyfit(x, s.values, 1)[0]) * 10)
        tds = np.array(tds)
        dx = delta[(radio, cn, pn)].reindex(sub.nombre).values
        filas.append({"datos": dn, "ventana": vn, "estimador": estim, "inicio": ini, "radio": radio,
                      "cultivo": cn, "periodo": pn, "estaciones": conj, "n": len(sub),
                      "rho": stats.spearmanr(dx, tds)[0],
                      "parcial": parcial(dx, tds, np.c_[sub.lat, sub.lon])})
    r = pd.DataFrame(filas)
    r.to_csv("analisis/23_multiverso.csv", index=False)

    def resumen(m, etiqueta):
        q = r[m]
        print(f"{etiqueta:38s} n={len(q):4d}  ρ mediana {q.rho.median():+.2f} [{q.rho.quantile(.05):+.2f}, "
              f"{q.rho.quantile(.95):+.2f}]  ρ<0: {(q.rho < 0).mean():.0%}  ρ<−0.44 (p<.05): "
              f"{(q.rho < -0.44).mean():.0%} | parcial mediana {q.parcial.median():+.2f} "
              f"[{q.parcial.quantile(.05):+.2f}, {q.parcial.quantile(.95):+.2f}]  <0: {(q.parcial < 0).mean():.0%}")

    principal = r.ventana.isin(["OND", "ONDJFM"]) & (r.cultivo != "trigo")
    print()
    resumen(principal, "PRINCIPAL (OND+ONDJFM, cultivos verano)")
    resumen(principal & (r.ventana == "OND"), "  solo OND")
    resumen(principal & (r.ventana == "ONDJFM"), "  solo ONDJFM")
    resumen((r.ventana == "JFM") & (r.cultivo != "trigo"), "CONTROL ene-mar")
    resumen(r.ventana.isin(["OND", "ONDJFM"]) & (r.cultivo == "trigo"), "PLACEBO trigo")
    print("\nSensibilidad por decisión (PRINCIPAL, mediana de ρ):")
    for c in ("datos", "ventana", "estimador", "inicio", "radio", "cultivo", "periodo", "estaciones"):
        print(f"  {c:11s}", r[principal].groupby(c).rho.median().round(2).to_dict())


if __name__ == "__main__":
    main()
