"""Índice de intensificación: superficie sembrada con cultivos de verano (MAGyP) / superficie agrícola (MapBiomas), 100 km.

Antes este CSV se había generado a mano, sin script (revisión K, 3 de octubre); este script lo reproduce exactamente.
- Superficie agrícola (MapBiomas Col. 3, % del círculo): cultivos temporarios (19, 36, 18), pasturas (15), mosaico
  agricultura–pastura (21) y plantaciones forestales (9); media 1985-89 y 2015-19.
- Fracción sembrada (MAGyP, % del círculo, 19_fraccion_anual_verano.csv): campañas 1985/86-1989/90 y 2015/16-2019/20
  (el archivo indexa la campaña por su año final, de ahí el desplazamiento de un año).
- Índice = sembrada / agrícola × 100. No se calcula donde la superficie agrícola cubre < 5 % del círculo (índice
  inestable: Formosa, La Rioja, Paso de los Libres, Resistencia), así que queda para 16 estaciones.
Salida: analisis/26_intensificacion.csv
"""
import os

import numpy as np
import pandas as pd

AGRO = ["c19", "c36", "c18", "c15", "c21", "c9"]
MIN_AGRO = 5.0


def main():
    c = pd.read_csv("data/mapbiomas/composicion.csv").fillna(0.0)
    c = c[c.radio == 100].assign(nombre=lambda x: x.nombre.str.strip())
    c["agro"] = c[[k for k in AGRO if k in c]].sum(axis=1) * 100
    fr = pd.read_csv("analisis/19_fraccion_anual_verano.csv", index_col="anio")
    fr.columns = fr.columns.str.strip()
    d = pd.read_csv("analisis/26_mapbiomas_delta.csv")
    d = d[d.radio == 100].assign(nombre=lambda x: x.nombre.str.strip())
    filas = []
    for _, e in d.iterrows():
        a85 = c[(c.nombre == e.nombre) & c.anio.between(1985, 1989)].agro.mean()
        a15 = c[(c.nombre == e.nombre) & c.anio.between(2015, 2019)].agro.mean()
        m85, m15 = fr[e.nombre].loc[1986:1990].mean(), fr[e.nombre].loc[2016:2020].mean()
        i85 = m85 / a85 * 100 if a85 >= MIN_AGRO else np.nan
        i15 = m15 / a15 * 100 if a15 >= MIN_AGRO else np.nan
        filas.append({"nombre": e.nombre, "agro85": a85, "agro15": a15, "int85": i85, "int15": i15,
                      "d_magyp": m15 - m85, "lat": e.lat, "lon": e.lon, "td_ond": e.td_ond,
                      "td_ondjfm": e.td_ondjfm, "td_jfm": e.td_jfm, "d_pastura": e.d_pastura,
                      "d_cultivos": e.d_cultivos, "d_perennes_herb": e.d_perennes_herb,
                      "d_int": i15 - i85, "d_agro": a15 - a85})
    out = pd.DataFrame(filas)
    out.to_csv("analisis/26_intensificacion.csv", index=False)
    print(f"{out.d_int.notna().sum()} estaciones con índice; sin índice: {list(out.nombre[out.d_int.isna()])}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
