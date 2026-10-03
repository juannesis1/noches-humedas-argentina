"""Frecuencia de noches con niebla (NOAA ISD completo: visibilidad y tiempo presente), oct-dic, 1980-2025.

Una noche tiene niebla si a las 06 o 09 UTC: código de tiempo presente manual (MW1) 40-49 (niebla) u 11-12 (niebla
baja), o visibilidad < 1000 m con depresión del punto de rocío ≤ 2 °C (excluye humo y bruma seca).
Si el secado de la zona intensificada es real, debería acompañarse de menos noches con niebla; si fuera solo un
artefacto del higrómetro, la niebla (observada por visibilidad y por el observador) no debería cambiar.
Se reporta la cobertura por década, porque la observación de tiempo presente cambia con la automatización.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402


def leer(sid):
    partes = []
    for f in sorted(glob.glob(f"data/isd_full/{sid}_*.csv")):
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("DATE", "VIS", "MW1", "TMP", "DEW"), dtype=str)
        except Exception:
            continue
        partes.append(d)
    if not partes:
        return None
    d = pd.concat(partes)
    d["DATE"] = pd.to_datetime(d.DATE)
    d = d[d.DATE.dt.hour.isin([6, 9]) & (d.DATE.dt.minute == 0) & d.DATE.dt.month.isin([10, 11, 12])]
    num = lambda s, esc: pd.to_numeric(s.str.split(",").str[0], errors="coerce").where(lambda x: x.abs() < 9999 * esc) / esc
    vis = pd.to_numeric(d.VIS.str.split(",").str[0], errors="coerce") if "VIS" in d else np.nan
    vis = vis.where(vis < 999999)
    t, td = num(d.TMP, 10), num(d.DEW, 10)
    mw = pd.to_numeric(d.MW1.str.split(",").str[0], errors="coerce") if "MW1" in d else pd.Series(np.nan, index=d.index)
    niebla_mw = mw.between(40, 49) | mw.between(11, 12)
    niebla_vis = (vis < 1000) & ((t - td) <= 2)
    obs = vis.notna() | mw.notna()
    out = pd.DataFrame({"fecha": d.DATE.dt.normalize(), "niebla": (niebla_mw | niebla_vis) & obs, "obs": obs})
    out = out[out.obs].groupby("fecha").niebla.any()
    return out


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    filas, cob = [], []
    for _, e in est.iterrows():
        s = leer(e.sid.replace("-", ""))
        if s is None:
            continue
        temp = s.index.year + 1
        g = pd.Series(s.values.astype(float), index=temp).groupby(level=0)
        n = g.size()
        frac = (g.mean() * 100)[n >= 46]
        frac = frac[(frac.index >= 1980) & (frac.index <= 2025)]
        for k, v in n.groupby((n.index // 10) * 10).mean().items():
            cob.append({"nombre": e.nombre.strip(), "decada": k, "noches": v})
        ok = len(frac) >= 30 and (frac.index <= 1989).sum() >= 5 and (frac.index >= 2016).sum() >= 5
        filas.append({"nombre": e.nombre.strip(), "lat": e.lat, "lon": e.lon, "niebla_media": frac.mean(),
                      "niebla_tend": sen(frac.index.values.astype(float), frac.values) * 10 if ok else np.nan,
                      "n_temp": len(frac)})
    r = pd.DataFrame(filas).merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    r.to_csv("analisis/34_niebla.csv", index=False)
    print("Noches oct-dic con dato (06 o 09 UTC), media por década:")
    print(pd.DataFrame(cob).pivot_table(index="decada", values="noches", aggfunc="median").round(0).T.to_string())
    pd.set_option("display.width", 200)
    print(r.sort_values("delta_soja_pp").round(2).to_string(index=False))
    q = r.dropna(subset=["delta_soja_pp", "niebla_tend"])
    a, b = q[q.delta_soja_pp > 20].niebla_tend.mean(), q[q.delta_soja_pp < 5].niebla_tend.mean()
    print(f"\nTendencia de noches con niebla (pp/déc): alta exp {a:+.2f}, baja {b:+.2f}; "
          f"ρ con cultivos {stats.spearmanr(q.delta_soja_pp, q.niebla_tend)[0]:+.2f} (n={len(q)})")


if __name__ == "__main__":
    main()
