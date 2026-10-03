"""Detección del jet en ERA5 (oct-mar) con tres criterios y validación contra radiosondeos.

A. Salio et al. (2002), "Chaco jet": en al menos una de las 4 horas sinópticas,
   (1) la isotaca de 12 m/s a 850 hPa es continua desde latitudes tropicales hasta 25°S: en cada fila
       de latitud entre LAT_INI y 25°S existe algún punto del corredor 66°W-57°W con |V|850 ≥ 12,
       v < 0 y |v| > |u|;
   (2) en algún punto dentro de la isotaca |V|850 - |V|700 ≥ 6 m/s;
   (3) en TODA el área de la caja encerrada por la isotaca el viento es del norte y |v| > |u|
       (excluye situaciones prefrontales con viento zonal dominante).
   Calibración (DEF 1979/80-1988/89, ERA5): LAT_INI = 20°S da 15.8 % de días con jet, frente al 17 %
   de Salio et al. (2002, ERA-15, 1979-93); 18°S da 12.3 %, 22°S 15.8 %, 15°S 0.8 % (el corredor a
   15-17°S está sobre la ladera andina en la grilla de 0.5°). Se adopta 20°S (límite tropical).
   Es una adaptación objetiva a grilla del criterio de isotaca de Salio et al. (2002).
   Evento (como en Salio et al. 2002): días que forman parte de rachas de ≥ 2 días consecutivos;
   su 17 % de los días DEF (1979-93) se refiere a días dentro de eventos ("jet_salio_evento").
B. Montini et al. (2019): a las 06 UTC en Santa Cruz (17.8°S, 63.2°W) y Mariscal Estigarribia
   (22.0°S, 60.6°W): |V|850 y cizalladura 850-700 > P75 mensual (1981-2010) y dirección NO-NE.
C. Criterio de radiosondeo en el punto de Resistencia (27.45°S, 59.05°W) a las 12 UTC,
   idéntico al de 08_jet_radiosondeo.py, para validar día a día contra IGRA.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))

PUNTOS = {"santa_cruz": (-17.8, -63.2), "mariscal": (-22.0, -60.6), "resistencia": (-27.45, -59.05)}


def abrir():
    import era5io                      # lee archivos completos y livianos (06c) juntos
    return era5io.abrir_pl()


LAT_INI = 20


def salio(ds):
    # desde 2 de octubre el viento faltante se bajó solo a 06 y 12 UTC (06c, uv2): se usan esas dos horas en TODO el
    # período para que la detección sea homogénea; recalibrar contra el 17 % de Salio et al. (2002)
    ds = ds.sel(time=ds.time.dt.hour.isin([6, 12]))
    corr = ds.sel(latitude=slice(-LAT_INI, -25), longitude=slice(-66, -57))
    u8, v8 = corr.u.sel(level=850), corr.v.sel(level=850)
    u7, v7 = corr.u.sel(level=700), corr.v.sel(level=700)
    w8 = np.hypot(u8, v8)
    w7 = np.hypot(u7, v7)
    nucleo = (w8 >= 12) & (v8 < 0) & (abs(v8) > abs(u8))
    filas = nucleo.any("longitude")                      # (time, latitude)
    continuo = filas.all("latitude")
    iso = w8 >= 12
    c3 = ((~iso) | ((v8 < 0) & (abs(v8) > abs(u8)))).all(["latitude", "longitude"])
    cz = ((w8 - w7) >= 6) & iso
    hora = (continuo & cz.any(["latitude", "longitude"]) & c3).compute()
    s = hora.to_series()
    dia = s.groupby(s.index.normalize()).any().rename("jet_salio")
    dia = dia.asfreq("D")                                  # huecos (abr-sep) quedan NaN
    racha = dia.fillna(False).astype(bool)
    grupo = (racha != racha.shift()).cumsum()
    largo = racha.groupby(grupo).transform("size")
    evento = (racha & (largo >= 2)).where(dia.notna())
    return pd.concat([dia, evento.rename("jet_salio_evento")], axis=1).dropna()


def punto(ds, lat, lon):
    p = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df = pd.DataFrame({
        "u850": p.u.sel(level=850).values, "v850": p.v.sel(level=850).values,
        "u700": p.u.sel(level=700).values, "v700": p.v.sel(level=700).values,
        "q850": p.q.sel(level=850).values * 1000}, index=pd.DatetimeIndex(p.time.values))
    df["w850"] = np.hypot(df.u850, df.v850)
    df["cz"] = df.w850 - np.hypot(df.u700, df.v700)
    d = (np.degrees(np.arctan2(-df.u850, -df.v850)) + 360) % 360
    df["norte"] = (d >= 292.5) | (d <= 45)
    return df


def p75(df):
    base = df[(df.index.year >= 1981) & (df.index.year <= 2010)]
    if base.empty:
        base = df
    pw = base.w850.groupby(base.index.month).quantile(0.75)
    pc = base.cz.groupby(base.index.month).quantile(0.75)
    return (df.w850.values >= pw.reindex(df.index.month).values) & \
           (df.cz.values >= pc.reindex(df.index.month).values) & df.norte.values


def main():
    ds = abrir()
    out = salio(ds)
    for nombre in ("santa_cruz", "mariscal"):
        df = punto(ds, *PUNTOS[nombre])
        df = df[df.index.hour == 6]
        out[f"jet_montini_{nombre}"] = pd.Series(p75(df), index=df.index.normalize())
    out["jet_montini"] = out[["jet_montini_santa_cruz", "jet_montini_mariscal"]].any(axis=1)
    r = punto(ds, *PUNTOS["resistencia"])
    r = r[r.index.hour == 12]
    out["res_bonner"] = pd.Series(((r.w850 >= 12) & (r.cz >= 6) & r.norte).values,
                                  index=r.index.normalize())
    out["res_p75"] = pd.Series(p75(r), index=r.index.normalize())
    out["q850_res"] = pd.Series(r.q850.values, index=r.index.normalize())
    out.index.name = "fecha"
    out.to_parquet("data/era5_jet_diario.parquet")
    print(out.mean(numeric_only=True).round(3))
    djf = out[out.index.month.isin([12, 1, 2])]
    temp = np.where(djf.index.month == 12, djf.index.year + 1, djf.index.year)
    cal = djf[(temp >= 1980) & (temp <= 1993)]
    print(f"Calibración Salio (DEF 1979/80-1992/93 disponibles, n={len(cal)} días, "
          f"{len(set(temp[(temp >= 1980) & (temp <= 1993)]))} veranos): "
          f"días con jet {cal.jet_salio.astype(float).mean():.1%}, días en eventos ≥2 d "
          f"{cal.jet_salio_evento.astype(float).mean():.1%} (Salio 2002: 17 %)")
    validar(out)


def validar(out):
    """Tabla de contingencia ERA5 vs. IGRA en Resistencia (12 UTC)."""
    a = pd.read_parquet("data/igra/proc/ARM00087155.parquet")
    a = a[a.index.hour == 12].dropna(subset=["u850", "v850", "u700", "v700"])
    w8 = np.hypot(a.u850, a.v850)
    cz = w8 - np.hypot(a.u700, a.v700)
    d = (np.degrees(np.arctan2(-a.u850, -a.v850)) + 360) % 360
    obs = pd.Series(((w8 >= 12) & (cz >= 6) & ((d >= 292.5) | (d <= 45))).values,
                    index=a.index.normalize())
    obs = obs[~obs.index.duplicated()]
    j = pd.concat([obs.rename("obs"), out.res_bonner.rename("era5")], axis=1).dropna().astype(bool)
    h = (j.obs & j.era5).sum()
    m = (j.obs & ~j.era5).sum()
    f = (~j.obs & j.era5).sum()
    c = (~j.obs & ~j.era5).sum()
    n = h + m + f + c
    pod = h / (h + m) if h + m else np.nan
    far = f / (h + f) if h + f else np.nan
    esperado = ((h + m) * (h + f) + (c + m) * (c + f)) / n
    hss = ((h + c) - esperado) / (n - esperado)
    print(f"Validación Resistencia 12 UTC (Bonner): n={n} aciertos={h} pérdidas={m} "
          f"falsas={f} POD={pod:.2f} FAR={far:.2f} HSS={hss:.2f}")
    pd.DataFrame([{"n": n, "aciertos": h, "perdidas": m, "falsas_alarmas": f, "POD": pod,
                   "FAR": far, "HSS": hss}]).to_csv("analisis/09_validacion_jet.csv", index=False)


if __name__ == "__main__":
    main()
