"""Descarga de ERA5 desde el Copernicus Climate Data Store (requiere ~/.cdsapirc).

1. Niveles de presión (detección del jet y transporte de humedad):
   u, v, q, t en 925, 850 y 700 hPa; 00, 06, 12 y 18 UTC (las 4 horas sinópticas de
   Salio et al. 2002); oct-mar; 1979-2025; 10°S-45°S, 75°W-45°W; grilla de 0.5°.
2. Superficie (comparación con estaciones y patrón espacial):
   t2m, d2m y presión de superficie a 06 y 09 UTC; oct-mar; misma región; grilla 0.25°.
Un archivo por mes y por producto (límite de costo del CDS); si ya existe, se saltea (permite reanudar).
"""
import os
import sys
import time

import cdsapi

SALIDA = "data/era5"
AREA = [-10, -75, -45, -45]          # N, O, S, E
MESES = ["01", "02", "03", "10", "11", "12"]
DIAS = [f"{d:02d}" for d in range(1, 32)]


def pedir(c, dataset, req, destino, intentos=10):
    """Si la cola del usuario está llena ("Number queued requests ... exceeded") espera 5 min y
    reintenta sin límite (no es un error del pedido); otros errores: hasta `intentos` con espera
    creciente (máx. 10 min). Así ningún mes queda salteado por saturación de la cola."""
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return
    tmp = f"{destino}.part{os.getpid()}"
    i = 0
    while i < intentos:
        try:
            c.retrieve(dataset, req, tmp)
            os.replace(tmp, destino)
            print(time.strftime("%H:%M"), "ok", destino, flush=True)
            return
        except Exception as e:  # errores transitorios del CDS (400/500 al consultar el trabajo)
            if os.path.exists(tmp):
                os.remove(tmp)
            if "queued requests" in str(e) or "rejected" in str(e):
                print(time.strftime("%H:%M"), f"cola llena, espero 5 min: {destino}", flush=True)
                time.sleep(300)
                continue
            i += 1
            print(time.strftime("%H:%M"), f"reintento {i} {destino}: {str(e)[:200]}", flush=True)
            time.sleep(min(60 * i, 600))
    print(time.strftime("%H:%M"), "FALLÓ", destino, flush=True)


def main(anios, producto):
    """producto: 'pl' (niveles de presión) o 'sfc' (superficie). Un proceso por producto
    para no exceder el límite de pedidos en cola por dataset del CDS."""
    os.makedirs(SALIDA, exist_ok=True)
    c = cdsapi.Client(quiet=True)
    for a in anios:
      for mes in MESES:
        if producto == "pl":
          pedir(c, "reanalysis-era5-pressure-levels", {
            "product_type": ["reanalysis"],
            "variable": ["u_component_of_wind", "v_component_of_wind", "specific_humidity",
                         "temperature"],
            "pressure_level": ["700", "850", "925"],
            "year": [str(a)], "month": [mes], "day": DIAS,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": AREA, "grid": [0.5, 0.5],
            "data_format": "netcdf", "download_format": "unarchived",
        }, f"{SALIDA}/pl_{a}_{mes}.nc")
        else:
          pedir(c, "reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure"],
            "year": [str(a)], "month": [mes], "day": DIAS,
            "time": ["06:00", "09:00"],
            "area": AREA,
            "data_format": "netcdf", "download_format": "unarchived",
        }, f"{SALIDA}/sfc_{a}_{mes}.nc")


if __name__ == "__main__":
    producto = sys.argv[1]
    anios = range(int(sys.argv[2]), int(sys.argv[3]) + 1) if len(sys.argv) > 3 else range(1979, 2026)
    if os.environ.get("REVERSO"):
        anios = list(anios)[::-1]
    main(anios, producto)
