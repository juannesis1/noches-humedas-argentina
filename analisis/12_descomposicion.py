"""Clasificación de noches por circulación y descomposición del cambio (Barry & Perry 1973;
Beck et al. 2007), con los tres términos reportados por separado.

Tipos de noche (con ERA5 a 06 UTC, la hora de la noche más cercana a nuestra métrica):
  JET       : día con jet del Chaco (criterio Salio, 09_jet_era5.py).
  ATLANTICO : sin jet y viento en 850 hPa en la caja 27-33°S, 57-61°W con dirección
              entre 22.5° y 135° (NE a SE) y |V| ≥ 3 m/s: aire de origen atlántico
              (Barros et al. 2008; Sun et al. 2017).
  NORTE     : sin jet, flujo del N/NO (292.5°-22.5°) débil o fuera del criterio de jet.
  SUR       : flujo del S/SO (135°-292.5°): post-frontal.
Para cada estación: X̄ = Σ fₖ x̄ₖ. Entre dos períodos (1980-2002 vs 2003-2025):
  ΔX = Σ x̄ₖ Δfₖ          (frecuencia / dinámico)
     + Σ fₖ Δx̄ₖ          (dentro del tipo / termodinámico)
     + Σ Δfₖ Δx̄ₖ         (interacción)
Además, versión continua: tendencia de x̄ₖ por tipo y de fₖ.
Se trabaja con anomalías diarias (climatología 1981-2010) para que el ciclo anual no se
confunda con la estacionalidad de los tipos.
Incertidumbre: bootstrap por temporadas (1000 remuestreos en bloque de temporadas).
Variables de entorno: P1, P2 ("1980-2002"), VENTANA ("10,11,12,1,2,3" u "10,11,12"), SUFIJO. Con ERA5
parcial se usa P1=1980-1989 y P2=2017-2025 (temporadas con niveles de presión completos).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402
from importlib import import_module  # noqa: E402

tend = import_module("03_tendencias")   # reutiliza la función de anomalías diarias 1981-2010

TIPOS = ["JET", "ATLANTICO", "NORTE", "SUR"]
VARS = ["t", "td", "tw", "q"]
RNG = np.random.default_rng(7)


def tipos_diarios():
    jet = pd.read_parquet("data/era5_jet_diario.parquet").jet_salio
    import era5io
    ds = era5io.abrir_pl()
    caja = ds.sel(level=850, latitude=slice(-27, -33), longitude=slice(-61, -57))
    caja = caja.where(caja.time.dt.hour == 6, drop=True)[["u", "v"]].mean(["latitude", "longitude"]).compute()
    u, v = caja.u.values, caja.v.values
    w = np.hypot(u, v)
    d = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    fechas = pd.DatetimeIndex(caja.time.values).normalize()
    tipo = np.where((d >= 22.5) & (d < 135) & (w >= 3), "ATLANTICO",
                    np.where((d >= 135) & (d < 292.5), "SUR", "NORTE"))
    t = pd.Series(tipo, index=fechas)
    t[jet.reindex(fechas).fillna(False).values] = "JET"
    return t


def descomponer(df, var, p1, p2):
    a, b = df[df.temporada.between(*p1)], df[df.temporada.between(*p2)]
    f1 = a.tipo.value_counts(normalize=True).reindex(TIPOS, fill_value=0)
    f2 = b.tipo.value_counts(normalize=True).reindex(TIPOS, fill_value=0)
    x1 = a.groupby("tipo")[var].mean().reindex(TIPOS)
    x2 = b.groupby("tipo")[var].mean().reindex(TIPOS)
    ok = x1.notna() & x2.notna()
    df_ = (f2 - f1)[ok]
    dx = (x2 - x1)[ok]
    return {"frecuencia": float((x1[ok] * df_).sum()), "dentro": float((f1[ok] * dx).sum()),
            "interaccion": float((df_ * dx).sum()), "total": float(b[var].mean() - a[var].mean())}


def bootstrap(df, var, p1, p2, n=1000):
    temporadas = df.temporada.unique()
    grupos = {t: g for t, g in df.groupby("temporada")}
    res = []
    for _ in range(n):
        c1 = [t for t in temporadas if p1[0] <= t <= p1[1]]
        c2 = [t for t in temporadas if p2[0] <= t <= p2[1]]
        s1 = RNG.choice(c1, size=len(c1))
        s2 = RNG.choice(c2, size=len(c2))
        partes = []
        for k, t in enumerate(np.concatenate([s1, s2])):
            if t in grupos:
                g = grupos[t].copy()
                g["temporada"] = p1[0] + k if k < len(s1) else p2[0] + k - len(s1)
                partes.append(g)
        res.append(descomponer(pd.concat(partes), var, p1, p2))
    return pd.DataFrame(res)


def main():
    tipo = tipos_diarios()
    tipo.rename("tipo").to_frame().to_parquet("data/tipos_noche.parquet")
    print("Frecuencia de tipos por década:")
    temp = np.where(tipo.index.month >= 10, tipo.index.year + 1, tipo.index.year)
    print(pd.crosstab((temp // 10) * 10, tipo.values, normalize="index").round(3))
    est = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    est = est[(est.temporada == "ONDJFM") & est.incluida & ~est.nombre.str.contains("CERES")]
    leer = lambda k, x: tuple(int(v) for v in os.environ.get(k, x).split("-"))
    p1, p2 = leer("P1", "1980-2002"), leer("P2", "2003-2025")
    meses = [int(m) for m in os.environ.get("VENTANA", "10,11,12,1,2,3").split(",")]
    print(f"Períodos {p1} vs {p2}, meses {meses}")
    filas = []
    for _, e in est.iterrows():
        d = pd.read_parquet(f"data/noches_ajustadas_pares_laxo/{e.sid}.parquet")
        d = d[d.index.month.isin(meses)]
        # anomalías respecto de la climatología diaria: evita que el ciclo anual se mezcle
        # con la frecuencia estacional de cada tipo de circulación
        d = tend.anomalias(d)
        d["tipo"] = tipo.reindex(d.index).values
        d = d.dropna(subset=["tipo"])
        for v in VARS:
            r = descomponer(d, v, p1, p2)
            b = bootstrap(d, v, p1, p2, n=300)
            fila = {"nombre": e.nombre, "lat": e.lat, "lon": e.lon, "var": v, **r}
            for k in ("frecuencia", "dentro", "interaccion"):
                fila[f"{k}_ic95"] = f"[{b[k].quantile(0.025):+.2f}, {b[k].quantile(0.975):+.2f}]"
            filas.append(fila)
        print(e.nombre, flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(f"analisis/12_descomposicion{os.environ.get('SUFIJO', '')}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[res["var"] == "td"].round(2).sort_values("lon").to_string(index=False))
    # resumen por grupos de expansión (umbral único > 20 / < 5 pp) y relación con cultivos
    from scipy import stats
    uso = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv").assign(nombre=lambda x: x.nombre.str.strip())
    q = res[res["var"] == "td"].assign(nombre=lambda x: x.nombre.str.strip()).merge(
        uso[["nombre", "delta_soja_pp"]], on="nombre").dropna(subset=["delta_soja_pp"])
    al, ba = q[q.delta_soja_pp > 20], q[q.delta_soja_pp < 5]
    lin = (f"ERA5 tipos {p1}-{p2} meses {meses}: alta total {al.total.mean():+.2f} = frec {al.frecuencia.mean():+.2f} "
           f"+ dentro {al.dentro.mean():+.2f} | baja {ba.total.mean():+.2f} = {ba.frecuencia.mean():+.2f} + "
           f"{ba.dentro.mean():+.2f} | ρ(cultivos, dentro) {stats.spearmanr(q.delta_soja_pp, q.dentro)[0]:+.2f}, "
           f"ρ(cultivos, frec) {stats.spearmanr(q.delta_soja_pp, q.frecuencia)[0]:+.2f}")
    print(lin)
    open(f"analisis/12_resumen{os.environ.get('SUFIJO', '')}.txt", "w").write(lin + "\n")


if __name__ == "__main__":
    main()
