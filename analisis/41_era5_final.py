"""ERA5 completo (1980-2025): jet, transporte de humedad a 850 hPa y humedad en capas bajas por región.

1. Jet de Chaco (Salio et al. 2002, 06/12 UTC): frecuencia por temporada (oct-mar y oct-dic), Sen + Mann-Kendall
   (Hamed-Rao) y por década; habilidad contra los sondeos de Resistencia (Bonner, 12 UTC) por subperíodo, porque la
   detección del jet en ERA5 no es homogénea en el tiempo.
2. Transporte meridional de humedad a 850 hPa (−q·v, positivo hacia el sur; g kg⁻¹ m s⁻¹) a través de 25°S y 30°S
   entre 66 y 57°W, media de 06 y 12 UTC, por temporada oct-dic; tendencia y descomposición en parte dinámica
   (v' con q climatológica) y termodinámica (q' con v climatológico).
3. q a 850 hPa (06 UTC) en la Pampa intensificada (30-36°S, 65-59°W) y en el litoral NE (25-30°S, 59-54°W), oct-dic.
Salidas: analisis/41_resumen.txt y analisis/41_series.csv (para la figura S7).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import mk_hamed_rao, sen  # noqa: E402

L = []


def temporada(idx):
    return np.where(idx.month >= 10, idx.year + 1, idx.year)


def tend(s):
    s = s.dropna()
    x = s.index.values.astype(float)
    return sen(x, s.values) * 10, mk_hamed_rao(x, s.values)[1]


def main():
    series = {}
    # 1. jet
    j = pd.read_parquet("data/era5_jet_diario.parquet")
    for vn, meses in (("ONDJFM", [10, 11, 12, 1, 2, 3]), ("OND", [10, 11, 12])):
        s = j.jet_salio.astype(float)
        s = s[s.index.month.isin(meses)]
        f = s.groupby(temporada(s.index)).mean()
        f = f[(f.index >= 1980) & (f.index <= 2025)] * 100
        series[f"jet_{vn}"] = f
        t, p = tend(f)
        dec = f.groupby((f.index // 10) * 10).mean().round(1).to_dict()
        L.append(f"Jet Chaco ERA5 {vn}: media {f.mean():.1f} % de días; tendencia {t:+.2f} pp/déc (p {p:.3f}); por década {dec}")
    a = pd.read_parquet("data/igra/proc/ARM00087155.parquet")
    a = a[a.index.hour == 12].dropna(subset=["u850", "v850", "u700", "v700"])
    w8 = np.hypot(a.u850, a.v850)
    d = (np.degrees(np.arctan2(-a.u850, -a.v850)) + 360) % 360
    ob = pd.Series(((w8 >= 12) & (w8 - np.hypot(a.u700, a.v700) >= 6) & ((d >= 292.5) | (d <= 45))).values,
                   index=a.index.normalize())
    ob = ob[~ob.index.duplicated()]
    jj = pd.concat([ob.rename("obs"), j.res_bonner.rename("era5")], axis=1).dropna().astype(bool)
    jj = jj[jj.index.month.isin([10, 11, 12, 1, 2, 3])]
    tt = temporada(jj.index)
    for a_, b_ in ((1980, 1995), (1996, 2010), (2011, 2025)):
        q = jj[(tt >= a_) & (tt <= b_)]
        h, m, f_ = (q.obs & q.era5).sum(), (q.obs & ~q.era5).sum(), (~q.obs & q.era5).sum()
        L.append(f"  Resistencia {a_}-{b_}: obs {q.obs.mean():.1%}, ERA5 {q.era5.mean():.1%}, POD {h / (h + m):.2f}, "
                 f"FAR {f_ / (h + f_):.2f} (n {len(q)})")
    fo = jj.obs.groupby(tt).mean() * 100
    fe = jj.era5.groupby(tt).mean() * 100
    series["jet_res_obs"], series["jet_res_era5"] = fo, fe
    L.append(f"  Resistencia, tendencia de la frecuencia oct-mar: obs {tend(fo)[0]:+.2f} pp/déc (p {tend(fo)[1]:.2f}); "
             f"ERA5 {tend(fe)[0]:+.2f} (p {tend(fe)[1]:.2f})")
    # 2 y 3. transporte y humedad
    import era5io
    ds = era5io.abrir_pl()
    ds = ds.sel(time=ds.time.dt.hour.isin([6, 12]) & ds.time.dt.month.isin([10, 11, 12]))
    for lat in (25, 30):
        c = ds.sel(level=850, longitude=slice(-66, -57)).sel(latitude=-lat, method="nearest")
        qv = (-(c.q * 1000) * c.v).mean("longitude").compute().to_series()
        qq = (c.q * 1000).mean("longitude").compute().to_series()
        vv = c.v.mean("longitude").compute().to_series()
        dia = lambda s: s.groupby(s.index.normalize()).mean()
        qv, qq, vv = dia(qv), dia(qq), dia(vv)
        ser = lambda s: s.groupby(temporada(s.index)).mean().loc[1980:2025]
        F, Q, V = ser(qv), ser(qq), ser(vv)
        dinam = -(Q.mean() * V)                              # v cambia, q fija
        termo = -(Q * V.mean())                              # q cambia, v fijo
        series[f"flujo{lat}"] = F
        t, p = tend(F)
        L.append(f"Transporte −qv a 850 hPa, {lat}°S (66-57°W), oct-dic: media {F.mean():.1f} g kg⁻¹ m s⁻¹; "
                 f"tendencia {t:+.2f}/déc (p {p:.3f}; {t / F.mean():+.1%}/déc); dinámica {tend(dinam)[0]:+.2f}, "
                 f"termodinámica {tend(termo)[0]:+.2f}; v {tend(V)[0]:+.2f} m s⁻¹/déc, q {tend(Q)[0]:+.3f} g kg⁻¹/déc")
    q06 = ds.q.sel(level=850).where(ds.time.dt.hour == 6, drop=True) * 1000
    for nombre, la, lo in (("Pampa", (-30, -36), (-65, -59)), ("NE", (-25, -30), (-59, -54))):
        s = q06.sel(latitude=slice(*la), longitude=slice(*lo)).mean(["latitude", "longitude"]).compute().to_series()
        s = s.groupby(temporada(s.index)).mean().loc[1980:2025]
        series[f"q850_{nombre}"] = s
        t, p = tend(s)
        L.append(f"q850 06 UTC oct-dic {nombre}: tendencia {t:+.3f} g kg⁻¹/déc (p {p:.3f}); "
                 f"1980-2002 {s.loc[:2002].mean():.2f} → 2003-2025 {s.loc[2003:].mean():.2f}")
    pd.DataFrame(series).to_csv("analisis/41_series.csv", index_label="temporada")
    open("analisis/41_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
