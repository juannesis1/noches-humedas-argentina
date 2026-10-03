"""Series por temporada cálida, selección de estaciones y tendencias 1980-2025.

Criterios (fijados antes de mirar resultados):
- Temporada válida: ≥70 % de noches con dato (ONDJFM = 182 noches; DEF = 90).
- Estación incluida: ≥80 % de temporadas válidas en 1980-2025 (≥37 de 46) y ≥7 válidas
  en cada extremo (1980-1989 y 2016-2025), para que los extremos no queden sin datos.
- Noche húmeda: Tw nocturna ≥ P90 de la estación (1981-2010, temporada ONDJFM).
Tendencias: Sen + Mann-Kendall modificado (Hamed & Rao 1998); significancia de campo con
FDR (α=0.10, Wilks 2016).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import fdr, mk_hamed_rao, sen  # noqa: E402

INICIO, FIN = 1980, 2025
VARS = ["t", "td", "tw", "q", "hr", "dpd"]


CRITERIO = os.environ.get("CRITERIO", "anomalias")


def anomalias(d):
    """Resta la climatología diaria 1981-2010 (media móvil centrada de 31 días)."""
    base = d[(d.temporada >= 1981) & (d.temporada <= 2010)]
    doy = base.index.dayofyear
    clim = base[VARS].groupby(doy).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True,
                                                                      min_periods=10).mean()
    clim = clim.iloc[15:-15]
    clim.index = range(1, 367)
    a = d.copy()
    a[VARS] = d[VARS].values - clim.loc[d.index.dayofyear].values
    return a


def temporadas(d, meses, n_noches):
    d = d[d.index.month.isin(meses)]
    if CRITERIO == "anomalias":
        a = anomalias(d)
        g = a.groupby("temporada")
        med = g[VARS].mean()
        n = g.size()
        # cada mes con al menos 30 % de noches, y la temporada con al menos 50 %
        pm = a.groupby(["temporada", a.index.month]).size().unstack().reindex(columns=meses)
        mes_ok = (pm.fillna(0) >= 0.3 * 30).all(axis=1)
        med["n"] = n
        med["valida"] = (n >= 0.5 * n_noches) & mes_ok.reindex(med.index).fillna(False)
    else:
        g = d.groupby("temporada")
        n = g.size()
        med = g[VARS].mean()
        med["n"] = n
        med["valida"] = n >= 0.7 * n_noches
    return med, d


def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"]).set_index("sid")
    filas, series = [], []
    for f in sorted(glob.glob(f"{os.environ.get('DIR_NOCHES', 'data/noches')}/*.parquet")):
        sid = os.path.basename(f)[:-8]
        d = pd.read_parquet(f)
        for nombre_t, meses, nn in (("ONDJFM", [10, 11, 12, 1, 2, 3], 182), ("DEF", [12, 1, 2], 90)):
            med, dd = temporadas(d, meses, nn)
            base = dd[(dd.temporada >= 1981) & (dd.temporada <= 2010)]
            if len(base) < 1000:
                continue
            p90 = np.nanpercentile(base.tw, 90)
            med["frac_humedas"] = dd.assign(h=dd.tw >= p90).groupby("temporada").h.mean() * 100
            med = med.loc[(med.index >= INICIO) & (med.index <= FIN)]
            val = med[med.valida]
            MINV = int(os.environ.get("MIN_VALIDAS", 37)); MINE = int(os.environ.get("MIN_EXTREMOS", 7))
            ok = (len(val) >= MINV and (val.index <= 1989).sum() >= MINE and (val.index >= 2016).sum() >= MINE)
            fila = {"sid": sid, "nombre": info.loc[sid, "nombre"], "lat": info.loc[sid, "lat"],
                    "lon": info.loc[sid, "lon"], "elev": info.loc[sid, "elev"], "temporada": nombre_t,
                    "n_validas": len(val), "incluida": ok, "tw_p90": round(p90, 2)}
            x = val.index.values.astype(float)
            for v in VARS + ["frac_humedas"]:
                y = val[v].values
                fila[f"{v}_sen"] = sen(x, y) * 10
                fila[f"{v}_p"] = mk_hamed_rao(x, y)[1]
            filas.append(fila)
            s = val[VARS + ["frac_humedas"]].copy()
            s["sid"], s["temp_tipo"] = sid, nombre_t
            series.append(s.reset_index())
    res = pd.DataFrame(filas)
    for tt in ("ONDJFM", "DEF"):
        m = (res.temporada == tt) & res.incluida
        for v in VARS + ["frac_humedas"]:
            res.loc[m, f"{v}_fdr"] = fdr(res.loc[m, f"{v}_p"].values)
    res.to_csv(f"analisis/03_tendencias_{CRITERIO}{os.environ.get('SUFIJO', '')}.csv", index=False)
    pd.concat(series).to_csv(f"analisis/03_series_temporada_{CRITERIO}{os.environ.get('SUFIJO', '')}.csv", index=False)
    inc = res[(res.temporada == "ONDJFM") & res.incluida].sort_values("lat")
    pd.set_option("display.width", 250)
    cols = ["nombre", "lat", "lon", "elev", "n_validas", "t_sen", "td_sen", "tw_sen", "q_sen",
            "hr_sen", "frac_humedas_sen", "t_fdr", "td_fdr", "tw_fdr", "frac_humedas_fdr"]
    print(f"[{CRITERIO}] Estaciones incluidas (ONDJFM): {len(inc)} de {(res.temporada == 'ONDJFM').sum()}")
    print(inc[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
