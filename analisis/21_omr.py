"""Prueba decisiva: observación menos reanálisis (OMR) para el secado nocturno de oct-dic.

ERA5 usa una cobertura del suelo fija y no representa el reemplazo de pasturas por cultivos.
Si el secado de las estaciones es local, ERA5 debería retener solo una fracción (≲ 20 % por el
peso de la observación en la 2D-OI de HR; Simmons et al. 2010) más la retroalimentación vía
humedad del suelo; si es de circulación, ERA5 debería reproducirlo. Ver NOTAS (tanda 10).
Para cada estación (Td nocturna homogeneizada, ventana VENTANA):
  - ERA5 Td en el punto de grilla más cercano: noche (06+09), 06 (hora de análisis 2D-OI) y
    09 UTC (fuera del análisis);
  - ERA5 Td en un anillo de 50-150 km (sin los puntos que "ven" la estación);
  - ERA5 q a 925 hPa (06 UTC), por encima de la capa nocturna en la llanura.
Tendencias de Sen (°C o g/kg por década) de las anomalías estacionales; OMR = estación − ERA5.
Luego: ¿qué relación tiene cada tendencia y cada OMR con la expansión de cultivos?
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

VENTANA = [int(m) for m in os.environ.get("VENTANA", "10,11,12").split(",")]
DIR = "data/noches_ajustadas_pares_laxo"
INI, FIN = 1980, 2025


def temporada(idx):
    return np.where(idx.month >= 10, idx.year + 1, idx.year)


def estacional(s):
    """Anomalía estacional de una serie diaria: clim. diaria 1981-2010 suavizada (31 d)."""
    s = s[s.index.month.isin(VENTANA)]
    temp = temporada(s.index)
    base = s[(temp >= 1981) & (temp <= 2010)]
    clim = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True,
                                                                      min_periods=10).mean()
    clim = clim.iloc[15:-15]
    clim.index = range(1, 367)
    a = s - clim.loc[s.index.dayofyear].values
    g = pd.Series(a.values, index=temp).groupby(level=0)
    out = g.mean()[g.size() >= 0.5 * 30.5 * len(VENTANA)]
    return out[(out.index >= INI) & (out.index <= FIN)]


def tend(s):
    s = s.dropna()
    return sen(s.index.values.astype(float), s.values) * 10 if len(s) >= 20 else np.nan


def distancia_km(lat0, lon0, lat, lon):
    la0, lo0, la, lo = map(np.radians, (lat0, lon0, lat, lon))
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    meses_sfc = sorted(glob.glob("data/era5/sfc_*.nc"))
    meses_pl = sorted(glob.glob("data/era5/pl_*.nc"))
    sfc = xr.open_mfdataset(meses_sfc, combine="by_coords").rename({"valid_time": "time"})
    import era5io
    pl = era5io.abrir_pl()
    nivel = "level"
    q925 = pl.q.sel({nivel: 925}).where(pl.time.dt.hour == 6, drop=True)
    lat2d, lon2d = np.meshgrid(sfc.latitude.values, sfc.longitude.values, indexing="ij")
    filas = []
    for _, e in est.iterrows():
        d = pd.read_parquet(f"{DIR}/{e.sid}.parquet")
        obs = estacional(d.td)
        caja = sfc.d2m.sel(latitude=slice(e.lat + 1.6, e.lat - 1.6),
                           longitude=slice(e.lon - 1.9, e.lon + 1.9)).load() - 273.15
        dist = distancia_km(e.lat, e.lon, *np.meshgrid(caja.latitude.values, caja.longitude.values,
                                                         indexing="ij"))
        punto = caja.sel(latitude=e.lat, longitude=e.lon, method="nearest")
        anillo = caja.where(xr.DataArray((dist >= 50) & (dist <= 150),
                                         dims=("latitude", "longitude"))).mean(["latitude", "longitude"])
        def diaria(da, horas):
            s = da.to_series()
            s = s[s.index.hour.isin(horas)]
            return s.groupby(s.index.normalize()).mean() if len(horas) > 1 else \
                pd.Series(s.values, index=s.index.normalize())
        e_noche = estacional(diaria(punto, [6, 9]))
        e06 = estacional(diaria(punto, [6]))
        e09 = estacional(diaria(punto, [9]))
        e_anillo = estacional(diaria(anillo, [6, 9]))
        qq = q925.sel(latitude=e.lat, longitude=e.lon, method="nearest").load().to_series() * 1000
        e_q = estacional(pd.Series(qq.values, index=qq.index.normalize()))
        comun = obs.index.intersection(e_noche.index)
        fila = {"nombre": e.nombre.strip(), "lat": e.lat, "lon": e.lon,
                "n_temp_comun": len(comun),
                "obs": tend(obs.loc[comun]), "era_noche": tend(e_noche.loc[comun]),
                "era_06": tend(e06.reindex(comun)), "era_09": tend(e09.reindex(comun)),
                "era_anillo": tend(e_anillo.reindex(comun)), "era_q925": tend(e_q.reindex(comun)),
                "r_interanual": obs.loc[comun].corr(e_noche.loc[comun])}
        fila["omr"] = fila["obs"] - fila["era_noche"]
        fila["omr_anillo"] = fila["obs"] - fila["era_anillo"]
        fila["fraccion_retenida"] = fila["era_noche"] / fila["obs"] if abs(fila["obs"]) > 0.05 else np.nan
        filas.append(fila)
    res = pd.DataFrame(filas)
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv")
    uso["nombre"] = uso.nombre.str.strip()
    res = res.merge(uso[["nombre", "delta_soja_pp"]], on="nombre", how="left")
    v = "".join(str(m) for m in VENTANA)
    res.to_csv(f"analisis/21_omr_{v}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"ERA5: {len(meses_sfc)} meses sfc, {len(meses_pl)} pl. Ventana {VENTANA}")
    print(res.sort_values("delta_soja_pp").round(3).to_string(index=False))
    a = res.dropna(subset=["delta_soja_pp"])
    print("\nRelación con Δcultivos (Spearman, n=%d):" % len(a))
    for c in ("obs", "era_noche", "era_06", "era_09", "era_anillo", "era_q925", "omr", "omr_anillo"):
        rho, p = stats.spearmanr(a.delta_soja_pp, a[c], nan_policy="omit")
        print(f"  {c:11s}: ρ {rho:+.2f} (p {p:.3f})   media {a[c].mean():+.3f}")
    alta = a[a.delta_soja_pp > 20]
    print(f"\nEstaciones de alta expansión (n={len(alta)}): obs {alta.obs.mean():+.3f}, "
          f"ERA5 {alta.era_noche.mean():+.3f}, retenido {alta.era_noche.mean() / alta.obs.mean():.0%}")


if __name__ == "__main__":
    main()
