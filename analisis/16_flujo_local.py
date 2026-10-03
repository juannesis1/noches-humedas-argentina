"""Prueba discriminante con datos de estación: ¿el secado nocturno es advección o local?

Alternativa de Yin et al. (2023): de noche la humedad en superficie depende más del
transporte que del acople suelo-atmósfera; si cambió la frecuencia de noches con flujo del
norte (húmedo) la Td podría bajar sin que el uso del suelo tenga nada que ver.

Cada noche se clasifica por el viento de la propia estación a 09 UTC:
  NORTE (292.5-67.5°), ESTE (67.5-157.5°), SUR (157.5-247.5°), OESTE (247.5-292.5°),
  CALMA (ws < 1 m/s). Dirección faltante o con bandera se descarta.
Descomposición por estación (Barry & Perry; Beck et al. 2007), con anomalías diarias de Td
respecto de la climatología 1981-2010 (igual que 03_tendencias):
  media_temporada = Σ_k f_k · a_k
  tendencia ≈ Σ ā_k · Δf_k  (frecuencia)  +  Σ f̄_k · Δa_k  (dentro del tipo)  + resto
Δ = pendiente de Sen ·10 (por década). Luego se correlaciona cada término con la expansión
de cultivos de verano (13_uso_suelo_verano_100km.csv): si el término "dentro del tipo"
conserva la relación y el de frecuencia no, el cambio de régimen de viento local no la explica.
Ventanas: ONDJFM, OND (donde está la señal) y JFM.
Advertencia: los cambios de anemómetro afectan sobre todo la fracción de calmas; por eso se
repite la cuenta sin la clase CALMA (se reparte en su sector de dirección).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen, parcial  # noqa: E402

DIR = os.environ.get("DIR_NOCHES", "data/noches_ajustadas_pares_laxo")
VENTANAS = {"ONDJFM": [10, 11, 12, 1, 2, 3], "OND": [10, 11, 12], "JFM": [1, 2, 3]}
MIN_NOCHES_TIPO = 5


def sector(wd, ws, con_calma):
    s = pd.Series(np.nan, index=wd.index, dtype=object)
    ok = (wd >= 0) & (wd <= 360)
    s[ok & ((wd >= 292.5) | (wd < 67.5))] = "NORTE"
    s[ok & (wd >= 67.5) & (wd < 157.5)] = "ESTE"
    s[ok & (wd >= 157.5) & (wd < 247.5)] = "SUR"
    s[ok & (wd >= 247.5) & (wd < 292.5)] = "OESTE"
    if con_calma:
        s[(ws >= 0) & (ws < 1)] = "CALMA"
    else:
        s[ok & (ws >= 0) & (ws < 1) & (wd == 0)] = np.nan   # calma sin dirección
    return s


def anomalia_td(d):
    base = d[(d.temporada >= 1981) & (d.temporada <= 2010)]
    clim = base.td.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True,
                                                                      min_periods=10).mean()
    clim = clim.iloc[15:-15]
    clim.index = range(1, 367)
    return d.td - clim.loc[d.index.dayofyear].values


def descomponer(d, meses, con_calma):
    d = d[d.index.month.isin(meses)].copy()
    d["tipo"] = sector(d.wd09, d.ws09, con_calma)
    d = d.dropna(subset=["tipo", "a"])
    d = d[(d.temporada >= 1980) & (d.temporada <= 2025)]
    n = d.groupby("temporada").size()
    validas = n[n >= 0.5 * 30.5 * len(meses)].index
    d = d[d.temporada.isin(validas)]
    if len(validas) < 30:
        return None
    f = pd.crosstab(d.temporada, d.tipo, normalize="index")
    cnt = pd.crosstab(d.temporada, d.tipo)
    a = d.groupby(["temporada", "tipo"]).a.mean().unstack().where(cnt >= MIN_NOCHES_TIPO)
    x = f.index.values.astype(float)
    total = d.groupby("temporada").a.mean()
    out = {"tend_total": sen(x, total.values) * 10, "n_temp": len(validas)}
    frec = dentro = 0.0
    for k in f.columns:
        df_k = sen(x, f[k].values) * 10
        ok = a[k].notna()
        da_k = sen(x[ok.values], a[k][ok].values) * 10 if ok.sum() >= 20 else 0.0
        ab = np.average(a[k][ok], weights=cnt[k][ok]) if ok.any() else 0.0
        frec += ab * df_k
        dentro += f[k].mean() * da_k
        out[f"f_{k}"] = f[k].mean() * 100
        out[f"df_{k}"] = df_k * 100          # pp/década
        out[f"da_{k}"] = da_k
    out["frecuencia"] = frec
    out["dentro"] = dentro
    out["resto"] = out["tend_total"] - frec - dentro
    return out




def main():
    info = pd.read_fwf("data/hadisd/station_fullinfo.txt", header=None,
                       colspecs=[(0, 12), (13, 43), (43, 51), (51, 60), (60, 68)],
                       names=["sid", "nombre", "lat", "lon", "elev"])
    info["nombre"] = info.nombre.str.strip()
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv")
    filas = []
    for fpath in sorted(glob.glob(f"{DIR}/*.parquet")):
        sid = os.path.basename(fpath)[:-8]
        nom = info.loc[info.sid == sid, "nombre"].iloc[0]
        if nom not in set(uso.nombre):
            continue
        d = pd.read_parquet(fpath)
        d["a"] = anomalia_td(d)
        for con_calma in (True, False):
            for v, meses in VENTANAS.items():
                r = descomponer(d, meses, con_calma)
                if r is None:
                    continue
                filas.append({"nombre": nom, "ventana": v, "calma": con_calma, **r})
    res = pd.DataFrame(filas).merge(uso[["nombre", "lat", "lon", "delta_soja_pp"]], on="nombre")
    res.to_csv("analisis/16_flujo_local.csv", index=False)
    pd.set_option("display.width", 250)
    for con_calma in (True, False):
        print(f"\n===== {'con' if con_calma else 'sin'} clase CALMA =====")
        for v in VENTANAS:
            s = res[(res.ventana == v) & (res.calma == con_calma)]
            print(f"\n--- {v}  (n={len(s)})  medias °C/déc: total {s.tend_total.mean():+.3f}  "
                  f"frecuencia {s.frecuencia.mean():+.3f}  dentro {s.dentro.mean():+.3f}  "
                  f"resto {s.resto.mean():+.3f}")
            for c in ("tend_total", "frecuencia", "dentro"):
                rho, p = stats.spearmanr(s.delta_soja_pp, s[c])
                rp, pp = parcial(s.delta_soja_pp.values, s[c].values, np.c_[s.lat, s.lon])
                print(f"  Δcultivos vs {c:10s}: ρ {rho:+.2f} (p {p:.3f})  parcial lat/lon {rp:+.2f} (p {pp:.3f})")
            dfc = [c for c in s.columns if c.startswith("df_")]
            print("  cambio de frecuencia medio (pp/déc):",
                  " ".join(f"{c[3:]} {s[c].mean():+.2f}" for c in dfc))
    s = res[(res.ventana == "OND") & res.calma].sort_values("delta_soja_pp")
    print("\nOND por estación (con calma):")
    print(s[["nombre", "delta_soja_pp", "tend_total", "frecuencia", "dentro", "resto",
             "f_NORTE", "df_NORTE", "da_NORTE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
