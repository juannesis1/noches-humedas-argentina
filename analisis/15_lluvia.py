"""Control de la lluvia (CHIRPS v2, mensual, 0.05°) sobre el resultado de uso del suelo.

Si en las zonas de expansión agrícola llovió menos en oct-dic, el secado nocturno podría ser
climático. Para cada estación argentina: precipitación media en 50 km, tendencia de Sen de
la lluvia de oct-dic y de oct-mar (temporadas 1982-2025; CHIRPS empieza en ene-1981).
Luego: (1) ¿la tendencia de lluvia se relaciona con la expansión de cultivos?;
(2) correlación parcial Td ~ Δcultivos controlando lat, lon y tendencia de lluvia.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen, parcial  # noqa: E402




def main():
    ds = xr.open_dataset("data/chirps/chirps_mensual.nc", decode_times=False)
    meses = pd.date_range("1960-01-01", periods=int(ds.T.max()) + 1, freq="MS")
    fechas = meses[np.floor(ds.T.values).astype(int)]
    pr = ds.precipitation.assign_coords(T=fechas)
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv")
    mens = pd.read_csv("analisis/14_uso_suelo_mensual.csv")  # solo para referencia
    filas = []
    for _, e in uso.iterrows():
        cerca = pr.sel(Y=slice(e.lat - 0.45, e.lat + 0.45), X=slice(e.lon - 0.55, e.lon + 0.55))
        s = cerca.mean(["X", "Y"]).to_series()
        s.index = pd.DatetimeIndex(s.index)
        temp = np.where(s.index.month >= 10, s.index.year + 1, s.index.year)
        df = pd.DataFrame({"pr": s.values, "mes": s.index.month, "temp": temp})
        fila = {"nombre": e.nombre}
        for nombre, ms in (("ond", [10, 11, 12]), ("ondjfm", [10, 11, 12, 1, 2, 3])):
            t = df[df.mes.isin(ms)].groupby("temp").agg(pr=("pr", "sum"), n=("pr", "size"))
            t = t[(t.n == len(ms)) & t.index.to_series().between(1982, 2025)]
            x = t.index.values.astype(float)
            fila[f"pr_{nombre}_media"] = t.pr.mean()
            fila[f"pr_{nombre}_tend_pct_dec"] = sen(x, t.pr.values) * 10 / t.pr.mean() * 100
            fila[f"pr_{nombre}_p"] = mk_hamed_rao(x, t.pr.values)[1]
        filas.append(fila)
    p = pd.DataFrame(filas)
    d = uso.merge(p, on="nombre")
    d.to_csv("analisis/15_lluvia.csv", index=False)
    pd.set_option("display.width", 220)
    print(d[["nombre", "delta_soja_pp", "td_sen", "pr_ond_media", "pr_ond_tend_pct_dec", "pr_ond_p",
             "pr_ondjfm_tend_pct_dec"]].round(2).sort_values("delta_soja_pp").to_string(index=False))
    for c in ("pr_ond_tend_pct_dec", "pr_ondjfm_tend_pct_dec"):
        r, pv = stats.spearmanr(d.delta_soja_pp, d[c])
        print(f"Δcultivos vs {c}: ρ = {r:+.2f} (p {pv:.3f})")
        r, pv = stats.spearmanr(d.td_sen, d[c])
        print(f"tendencia Td vs {c}: ρ = {r:+.2f} (p {pv:.3f})")
    r, pv = parcial(d.delta_soja_pp.values, d.td_sen.values,
                    np.c_[d.lat, d.lon, d.pr_ondjfm_tend_pct_dec])
    print(f"Td ~ Δcultivos | lat, lon, lluvia oct-mar: r parcial = {r:+.2f} (p {pv:.3f})")
    r, pv = parcial(d.delta_soja_pp.values, d.td_sen.values,
                    np.c_[d.lat, d.lon, d.pr_ond_tend_pct_dec])
    print(f"Td ~ Δcultivos | lat, lon, lluvia oct-dic: r parcial = {r:+.2f} (p {pv:.3f})")


if __name__ == "__main__":
    main()
