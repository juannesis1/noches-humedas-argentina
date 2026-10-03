"""Cobertura del suelo observada por satélite (MapBiomas Argentina, Colección 3, 30 m, 1985-2025).

Responde a la revisión estricta (M8): la superficie sembrada (MAGyP) no mide el mecanismo propuesto
(pérdida de coberturas herbáceas perennes) y cuenta dos veces el doble cultivo. MapBiomas clasifica
cada píxel cada año (leyenda ARG-LegendCode-Col3):
  19 cultivos temporarios, 36 perennes, 18 agricultura; 15 pastura; 21 mosaico agricultura-pastura;
  12 herbáceas, 11 herbáceas inundables, 63/66/77 arbustales; 3/4/6 bosques; 9 silvicultura;
  24 urbano; 25 otras sin vegetación; 33 agua; 34 glaciares; 27 no observado; 0 sin dato.
Para cada año: se descarga el GeoTIFF público, se cuenta la composición dentro de 50, 100 y 150 km
de cada estación (distancia local con cos(lat)), se guarda y se borra el archivo (disco acotado).
Lectura a 1/4 de resolución (≈120 m, vecino más cercano): la fracción de clases en círculos de
50-150 km no cambia de forma apreciable y el tiempo baja ~16 veces.
Fracciones sobre píxeles de tierra observados (excluye 0, 27, 33, 34). Si menos del 60 % del círculo
cae dentro de Argentina (cobertura), la estación queda sin valor (Asunción, Montevideo, Rocha).
Uso: python analisis/25_mapbiomas.py [años...]   (por defecto: 1985-1989, 2015-2019 y luego el resto)
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

URL = ("https://storage.googleapis.com/mapbiomas-public/initiatives/argentina/lulc/collection_03/"
       "integration/integration-argentina_classification_{}.tif")
TMP = "data/mapbiomas/tmp_{}.tif"
SALIDA = "data/mapbiomas/composicion.csv"
RADIOS = (50, 100, 150)
FACTOR = 4
NO_TIERRA = {0, 27, 33, 34}


def estaciones():
    d = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    d = d[(d.temporada == "ONDJFM") & d.incluida & ~d.nombre.str.contains("CERES")]
    return d[["sid", "nombre", "lat", "lon"]].assign(nombre=lambda x: x.nombre.str.strip())


def procesar(anio, est):
    tif = TMP.format(anio)
    if not os.path.exists(tif):
        subprocess.run(["curl", "-s", "-f", "--retry", "5", "-o", tif + ".part", URL.format(anio)], check=True)
        os.replace(tif + ".part", tif)
    filas = []
    with rasterio.open(tif) as r:
        for _, e in est.iterrows():
            dlat = max(RADIOS) / 111.0
            dlon = dlat / np.cos(np.radians(e.lat))
            w = from_bounds(e.lon - dlon, e.lat - dlat, e.lon + dlon, e.lat + dlat, r.transform)
            w = w.round_offsets().round_lengths()
            alto, ancho = int(w.height // FACTOR), int(w.width // FACTOR)
            a = r.read(1, window=w, out_shape=(alto, ancho), boundless=True, fill_value=0,
                       resampling=rasterio.enums.Resampling.nearest)
            t = r.window_transform(w)
            xs = t.c + (np.arange(ancho) + 0.5) * t.a * FACTOR
            ys = t.f + (np.arange(alto) + 0.5) * t.e * FACTOR
            X, Y = np.meshgrid(xs, ys)
            d = np.hypot((X - e.lon) * 111.0 * np.cos(np.radians(e.lat)), (Y - e.lat) * 111.0)
            for rk in RADIOS:
                dentro = d <= rk
                vals = a[dentro]
                cobertura = np.mean(vals != 0)
                tierra = vals[~np.isin(vals, list(NO_TIERRA))]
                cls, n = np.unique(tierra, return_counts=True)
                fila = {"anio": anio, "nombre": e.nombre, "radio": rk, "cobertura": cobertura,
                        "n_tierra": len(tierra)}
                fila.update({f"c{c}": k / max(len(tierra), 1) for c, k in zip(cls, n)})
                filas.append(fila)
    os.remove(tif)
    return pd.DataFrame(filas)


def main():
    anios = [int(a) for a in sys.argv[1:]] or (list(range(1985, 1990)) + list(range(2015, 2020)) +
                                               [y for y in range(1990, 2026) if not 2015 <= y <= 2019])
    est = estaciones()
    hechos = set()
    if os.path.exists(SALIDA):
        hechos = set(pd.read_csv(SALIDA, usecols=["anio"]).anio.unique())
    for anio in anios:
        if anio in hechos:
            continue
        df = procesar(anio, est)
        previo = pd.read_csv(SALIDA) if os.path.exists(SALIDA) else None
        pd.concat([previo, df]).to_csv(SALIDA, index=False)
        print(anio, "ok", len(df), flush=True)


if __name__ == "__main__":
    main()
