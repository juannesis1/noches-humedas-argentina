"""Descarga liviana de los niveles de presión que faltan (reemplaza a 06b para 'pl').

Solo lo que usan los análisis: viento (u, v) a 700 y 850 hPa a 00/06/12/18 UTC (jet, tipos de circulación) y humedad
específica a 850 y 925 hPa a 06 y 12 UTC (Fig. 4, OMR, validación). Sin temperatura ni viento a 925.
Costos (estimate_costs, límite 60 000): viento 6 meses = 34 944 → una temporada (año calendario: ene-mar y oct-dic) por
pedido; humedad 3 años × 6 meses = 26 256 → bloques de hasta 6 años. Menos pedidos = menos esperas en la cola del CDS.
Archivos: pl_uv_AAAA.nc y pl_q_AAAA-AAAA.nc. La lectura conjunta con los archivos viejos la hace era5io.abrir_pl().
Variante uv2 (2 de octubre, 14:45): viento solo a 06 y 12 UTC → bloques de 3 años (costo 52 416) → 8 pedidos en vez
de 22; archivos pl_uv2_AAAA-AAAA.nc. La detección del jet en ERA5 pasa a usar 06 y 12 UTC en todo el período (se
recalibra contra Salio et al. 2002). Puede correr en dos procesos (REVERSO=1 recorre los bloques al revés).
Uso: python analisis/06c_descarga_pl_liviana.py uv|uv2|q
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
BASE = {"product_type": ["reanalysis"], "day": DIAS, "area": AREA, "grid": [0.5, 0.5],
        "data_format": "netcdf", "download_format": "unarchived"}
UV = {**BASE, "variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["700", "850"],
      "time": ["00:00", "06:00", "12:00", "18:00"]}
UV2 = {**BASE, "variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["700", "850"],
       "time": ["06:00", "12:00"]}
Q = {**BASE, "variable": ["specific_humidity"], "pressure_level": ["850", "925"], "time": ["06:00", "12:00"]}


def meses_completos():
    """Meses ya cubiertos por los archivos 'pl' completos (mensuales o trimestrales de 06/06b)."""
    tiene = set()
    for f in glob.glob(f"{SALIDA}/pl_*.nc"):
        if m := re.search(r"pl_(\d{4})_(\d\d)\.nc$", f):
            tiene.add((int(m.group(1)), m.group(2)))
        elif m := re.search(r"pl_(\d{4})_(\d\d)-(\d\d)\.nc$", f):
            tiene |= {(int(m.group(1)), f"{k:02d}") for k in range(int(m.group(2)), int(m.group(3)) + 1)}
    return tiene


def anios_faltantes():
    tiene = meses_completos()
    return sorted({y for y in range(1979, 2026) for m in MESES if (y, m) not in tiene})


def pedir(c, req, destino, intentos=10):
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return
    tmp = f"{destino}.part{os.getpid()}"
    i = 0
    while i < intentos:
        try:
            c.retrieve("reanalysis-era5-pressure-levels", req, tmp)
            xr.open_dataset(tmp).close()
            os.replace(tmp, destino)
            print(time.strftime("%H:%M"), "ok", destino, flush=True)
            return
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if any(k in str(e) for k in ("queued requests", "rejected", "temporarily limited")):
                print(time.strftime("%H:%M"), f"cola llena, espero 5 min: {destino}", flush=True)
                time.sleep(300)
                continue
            i += 1
            print(time.strftime("%H:%M"), f"reintento {i} {destino}: {str(e)[:200]}", flush=True)
            time.sleep(min(60 * i, 600))
    print(time.strftime("%H:%M"), "FALLÓ", destino, flush=True)


def main(tipo):
    c = cdsapi.Client(quiet=True)
    anios = anios_faltantes()
    if tipo == "uv":
        lista = [({**UV, "year": [str(y)], "month": MESES}, f"{SALIDA}/pl_uv_{y}.nc") for y in anios]
    elif tipo == "uv2":
        bloques = [anios[i:i + 3] for i in range(0, len(anios), 3)]
        lista = [({**UV2, "year": [str(y) for y in b], "month": MESES}, f"{SALIDA}/pl_uv2_{b[0]}-{b[-1]}.nc") for b in bloques]
        if os.environ.get("REVERSO"):
            lista = lista[::-1]
    else:
        bloques = [anios[i:i + 6] for i in range(0, len(anios), 6)]
        lista = [({**Q, "year": [str(y) for y in b], "month": MESES}, f"{SALIDA}/pl_q_{b[0]}-{b[-1]}.nc") for b in bloques]
    print(time.strftime("%H:%M"), f"{tipo}: {len(lista)} pedidos para {len(anios)} años", flush=True)
    for req, dest in lista:
        pedir(c, req, dest)


if __name__ == "__main__":
    main(sys.argv[1])
