"""Descarga de ERA5 por bloques grandes (reemplaza a 06_descarga_era5.py para lo que falta).

Diagnóstico (2 de octubre): el cuello de botella del CDS es la espera en cola por pedido, no el
tamaño. Límites de costo por pedido (estimate_costs): niveles de presión 60 000 (1 mes de nuestro pedido
= 17 856 → hasta 3 meses); superficie 121 000 (1 mes = 1 116 → hasta ~100 meses). Entonces:
  - superficie: un pedido por bloque de años completos (≤ 12 años × 6 meses) → sfc_bloque_AAAA-AAAA.nc
  - niveles de presión: un pedido por media temporada (ene-mar u oct-dic) → pl_AAAA_MM-MM.nc
Mismas variables, horas, área y grilla que 06_descarga_era5.py. Calcula qué meses faltan a partir de los
archivos existentes (mensuales o de bloque) y no repite. Espera ilimitada si la cola está llena.
Uso: python analisis/06b_descarga_bloques.py pl|sfc [REVERSO=1 recorre la lista al revés]
"""
import glob
import os
import re
import sys
import time

import cdsapi
import xarray as xr

SALIDA = "data/era5"
AREA = [-10, -75, -45, -45]
MESES = ["01", "02", "03", "10", "11", "12"]
DIAS = [f"{d:02d}" for d in range(1, 32)]
PL = {"product_type": ["reanalysis"],
      "variable": ["u_component_of_wind", "v_component_of_wind", "specific_humidity", "temperature"],
      "pressure_level": ["700", "850", "925"], "day": DIAS, "time": ["00:00", "06:00", "12:00", "18:00"],
      "area": AREA, "grid": [0.5, 0.5], "data_format": "netcdf", "download_format": "unarchived"}
SFC = {"product_type": ["reanalysis"],
       "variable": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure"],
       "day": DIAS, "time": ["06:00", "09:00"], "area": AREA, "data_format": "netcdf",
       "download_format": "unarchived"}


def meses_presentes(prod):
    tiene = set()
    for f in glob.glob(f"{SALIDA}/{prod}_*.nc"):
        m = re.search(rf"{prod}_(\d{{4}})_(\d\d)\.nc$", f)
        if m:
            tiene.add((int(m.group(1)), m.group(2)))
            continue
        m = re.search(rf"{prod}_(\d{{4}})_(\d\d)-(\d\d)\.nc$", f)
        if m:
            for mm in range(int(m.group(2)), int(m.group(3)) + 1):
                tiene.add((int(m.group(1)), f"{mm:02d}"))
            continue
        m = re.search(rf"{prod}_bloque_(\d{{4}})-(\d{{4}})\.nc$", f)
        if m:
            for y in range(int(m.group(1)), int(m.group(2)) + 1):
                tiene |= {(y, mm) for mm in MESES}
    return tiene


def pedidos(prod):
    tiene = meses_presentes(prod)
    falta = [(y, m) for y in range(1979, 2026) for m in MESES if (y, m) not in tiene]
    out = []
    if prod == "sfc":
        completos = sorted({y for y, _ in falta if all((y, m) not in tiene for m in MESES)})
        bloques, actual = [], []
        for y in completos:
            if actual and (y != actual[-1] + 1 or len(actual) == 12):
                bloques.append(actual)
                actual = []
            actual.append(y)
        if actual:
            bloques.append(actual)
        for b in bloques:
            out.append(({"year": [str(y) for y in b], "month": MESES}, f"sfc_bloque_{b[0]}-{b[-1]}.nc"))
        for y in sorted({y for y, _ in falta} - set(completos)):
            ms = [m for m in MESES if (y, m) not in tiene]
            for m in ms:
                out.append(({"year": [str(y)], "month": [m]}, f"sfc_{y}_{m}.nc"))
    else:
        for y in sorted({y for y, _ in falta}):
            for mitad in (["01", "02", "03"], ["10", "11", "12"]):
                ms = [m for m in mitad if (y, m) not in tiene]
                if not ms:
                    continue
                if len(ms) == 3:
                    out.append(({"year": [str(y)], "month": ms}, f"pl_{y}_{ms[0]}-{ms[-1]}.nc"))
                else:
                    out += [({"year": [str(y)], "month": [m]}, f"pl_{y}_{m}.nc") for m in ms]
    return out


def pedir(c, dataset, req, destino, intentos=10):
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return
    tmp = f"{destino}.part{os.getpid()}"
    i = 0
    while i < intentos:
        try:
            c.retrieve(dataset, req, tmp)
            xr.open_dataset(tmp).close()          # verifica que el NetCDF sea legible
            os.replace(tmp, destino)
            print(time.strftime("%H:%M"), "ok", destino, flush=True)
            return
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if "queued requests" in str(e) or "rejected" in str(e) or "temporarily limited" in str(e):
                print(time.strftime("%H:%M"), f"cola llena, espero 5 min: {destino}", flush=True)
                time.sleep(300)
                continue
            i += 1
            print(time.strftime("%H:%M"), f"reintento {i} {destino}: {str(e)[:200]}", flush=True)
            time.sleep(min(60 * i, 600))
    print(time.strftime("%H:%M"), "FALLÓ", destino, flush=True)


def main(prod):
    c = cdsapi.Client(quiet=True)
    dataset, base = (("reanalysis-era5-pressure-levels", PL) if prod == "pl"
                     else ("reanalysis-era5-single-levels", SFC))
    lista = pedidos(prod)
    if os.environ.get("REVERSO"):
        lista = lista[::-1]
    print(time.strftime("%H:%M"), f"{prod}: {len(lista)} pedidos", flush=True)
    for req, nombre in lista:
        pedir(c, dataset, {**base, **req}, f"{SALIDA}/{nombre}")


if __name__ == "__main__":
    main(sys.argv[1])
