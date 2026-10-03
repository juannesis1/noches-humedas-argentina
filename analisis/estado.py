"""Estado de todas las descargas y análisis en curso, con porcentaje de avance.
Uso: .venv/bin/python analisis/estado.py      (o en bucle: ver comando en el README de la respuesta)
"""
import glob
import os
import re
import subprocess

MESES = ["01", "02", "03", "10", "11", "12"]


def vivo(patron):
    return subprocess.run(["pgrep", "-f", patron], capture_output=True).returncode == 0


def barra(hecho, total, ancho=25):
    p = hecho / total if total else 0
    return f"[{'#' * int(p * ancho):<{ancho}}] {p:5.0%} ({hecho}/{total})"


def meses(prod):
    tiene = set()
    for f in glob.glob(f"data/era5/{prod}_*.nc"):
        if m := re.search(rf"{prod}_(\d{{4}})_(\d\d)\.nc$", f):
            tiene.add((m.group(1), m.group(2)))
        elif m := re.search(rf"{prod}_(\d{{4}})_(\d\d)-(\d\d)\.nc$", f):
            tiene |= {(m.group(1), f"{k:02d}") for k in range(int(m.group(2)), int(m.group(3)) + 1)}
        elif m := re.search(rf"{prod}_bloque_(\d{{4}})-(\d{{4}})\.nc$", f):
            tiene |= {(str(y), mm) for y in range(int(m.group(1)), int(m.group(2)) + 1) for mm in MESES}
    return tiene


def faltan_pl():
    return sorted({y for y in range(1979, 2026) for m in MESES if (str(y), m) not in meses("pl")})


estado = lambda p: "corriendo" if vivo(p) else "detenido"
print("=" * 70)
sfc = meses("sfc")
print(f"ERA5 superficie        {barra(len(sfc), 282)}  {estado('06b_descarga_bloques')}")
falt = faltan_pl()
uv = len(glob.glob("data/era5/pl_uv2_*.nc"))
nq = len(glob.glob("data/era5/pl_q_*.nc"))
# (línea "presión completo" eliminada: contaba solo el formato viejo; viento y humedad livianos completan el resto)
print(f"ERA5 presión: viento   {barra(uv, -(-len(falt) // 3))}  {estado('06c_descarga_pl_liviana.py uv2')}")
print(f"ERA5 presión: humedad  {barra(nq, -(-len(falt) // 6))}  {estado('06c_descarga_pl_liviana.py q')}")
print(f"NDVI satelital         {barra(len(glob.glob('data/ndvi/ndvi_*.nc')), 8101)}  {estado('ndvi_urls2.txt')}")
print(f"NOAA ISD (niebla)      {barra(len(glob.glob('data/isd_full/*.csv')), 1081)}  {estado('isd_urls.txt')}")
r = open("analisis/RESUMEN_PENDIENTES.txt").read() if os.path.exists("analisis/RESUMEN_PENDIENTES.txt") else ""
etapas = [("Niebla", "== NIEBLA"), ("NDVI", "== NDVI"), ("ERA5 final", "Listo:")]
print("Análisis encadenados:  " + "  ".join(f"{n}: {'✔' if k in r else '…'}" for n, k in etapas)
      + f"   ({estado('correr_pendientes.sh')})")
print("=" * 70)
