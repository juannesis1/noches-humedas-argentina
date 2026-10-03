"""Resumen gráfico para la IJC: imagen de 50 × 60 mm (ancho × alto), legible a ese tamaño.

Mapa de la tendencia de Td nocturna de oct-dic (1982-2025, script 37) en las 23 estaciones: el dipolo entre la Pampa
intensificada (más seca) y el litoral NE (más húmedo). Salida: envio/ijc/graphical_abstract.png (600 dpi) y .tif.
"""
import importlib
import os
import sys

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
fp = importlib.import_module("figuras_paper")


def main():
    r = pd.read_csv("analisis/37_resumen_estaciones.csv")
    mm = 1 / 25.4
    fig = plt.figure(figsize=(50 * mm, 60 * mm), dpi=600)
    ax = fig.add_axes([0.02, 0.17, 0.96, 0.70], projection=ccrs.PlateCarree())
    fp.mapa_base(ax)
    ax.set_extent([-67.8, -53.5, -39.5, -23.5], crs=ccrs.PlateCarree())
    norma = mpl.colors.TwoSlopeNorm(vmin=-0.45, vcenter=0, vmax=0.45)
    ax.scatter(r.lon, r.lat, c=r.td.clip(-0.45, 0.45), cmap=fp.CMAP_HUM, norm=norma, s=16, edgecolors=fp.INK,
               linewidths=0.35, transform=ccrs.PlateCarree(), zorder=5)
    kw = dict(transform=ccrs.PlateCarree(), fontsize=5.2, ha="center", zorder=6, fontweight="bold")
    ax.text(-62.6, -36.9, "Pampas:\ndrier", color=fp.DRY, **kw)
    ax.text(-61.4, -25.9, "Northeast:\nmore humid", color=fp.WET, **kw)
    fig.text(0.5, 0.955, "Spring-night dew point", ha="center", va="top", fontsize=6.2, fontweight="bold")
    fig.text(0.5, 0.905, "trend, Oct–Dec 1982–2025", ha="center", va="top", fontsize=5.4)
    cax = fig.add_axes([0.14, 0.10, 0.72, 0.03])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norma, cmap=fp.CMAP_HUM), cax=cax, orientation="horizontal",
                      extend="both", ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.4, pad=1)
    cb.outline.set_visible(False)
    fig.text(0.5, 0.015, "°C per decade", ha="center", fontsize=5.0)
    os.makedirs("envio/ijc", exist_ok=True)
    fig.savefig("envio/ijc/graphical_abstract.png", dpi=600)
    fig.savefig("envio/ijc/graphical_abstract.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
