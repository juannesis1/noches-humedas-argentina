#!/bin/bash
# Reproduce todos los resultados, figuras y el PDF del manuscrito a partir de los datos crudos ya descargados
# (ver DATOS.md para la descarga). Uso:
#   ./REPRODUCIR.sh              # todo
#   ./REPRODUCIR.sh desde 12     # retoma desde la etapa cuyo nombre empieza con "12"
# Cada paso escribe su salida en logs/<paso>.log; si un paso falla, el script se detiene y dice cuál.
# Tiempo aproximado en una laptop: 2-4 h (los bootstraps y la curva de especificación son lo más lento).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=analisis
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"
PY=.venv/bin/python
mkdir -p logs
DESDE="${2:-}"
activo=$([ -z "$DESDE" ] && echo 1 || echo 0)

paso() {   # paso <nombre> <comando...>
  local nombre="$1"; shift
  if [ "$activo" = 0 ] && [[ "$nombre" == "$DESDE"* ]]; then activo=1; fi
  [ "$activo" = 1 ] || return 0
  printf '%s  %-38s' "$(date +%H:%M)" "$nombre"
  if env "$@" > "logs/$nombre.log" 2>&1; then echo "ok"; else echo "FALLÓ (ver logs/$nombre.log)"; exit 1; fi
}

echo "== 1. Estaciones: auditoría y series nocturnas (06 y 09 UTC) y diurnas (15 y 18 UTC)"
paso 01_auditoria                $PY analisis/01_auditoria.py
paso 02_noches                   $PY analisis/02_noches.py

echo "== 2. Selección, homogeneización y tendencias"
paso 03_estricto                 CRITERIO=anomalias $PY analisis/03_tendencias.py
paso 03_laxo                     CRITERIO=anomalias MIN_VALIDAS=30 MIN_EXTREMOS=5 SUFIJO=_laxo $PY analisis/03_tendencias.py
paso 03_dominio_laxo             $PY -c "import pandas as pd; t=pd.read_csv('analisis/03_tendencias_anomalias_laxo.csv'); open('data/hadisd/dominio_laxo.txt','w').write('\n'.join(t[(t.temporada=='ONDJFM')&t.incluida].sid.astype(str))+'\n')"
paso 04_compuesta                $PY analisis/04_homogeneidad.py
paso 04b_pares                   $PY analisis/04b_homogeneidad_pares.py
paso 04b_pares_laxo              CANDIDATAS=analisis/03_tendencias_anomalias_laxo.csv SALIDA_HOM=analisis/04b_homogeneidad_pares_laxo.csv $PY analisis/04b_homogeneidad_pares.py
paso 05_ajuste_pares             HOM=pares $PY analisis/05_ajuste.py
paso 05_ajuste_compuesta         HOM=compuesta $PY analisis/05_ajuste.py
paso 05_ajuste_pares_laxo        HOM=pares_laxo FUENTE_HOM=analisis/04b_homogeneidad_pares_laxo.csv SALIDA_AJ=data/noches_ajustadas_pares_laxo $PY analisis/05_ajuste.py
paso 03_aj_pares                 CRITERIO=anomalias DIR_NOCHES=data/noches_ajustadas_pares SUFIJO=_aj_pares $PY analisis/03_tendencias.py
paso 03_aj_compuesta             CRITERIO=anomalias DIR_NOCHES=data/noches_ajustadas_compuesta SUFIJO=_aj_compuesta $PY analisis/03_tendencias.py
paso 03_laxo_aj                  CRITERIO=anomalias DIR_NOCHES=data/noches_ajustadas_pares_laxo MIN_VALIDAS=30 MIN_EXTREMOS=5 SUFIJO=_laxo_aj $PY analisis/03_tendencias.py
paso 02_dias                     HORAS=15,18 SALIDA_NOCHES=data/dias DOMINIO=data/hadisd/dominio_laxo.txt $PY analisis/02_noches.py
paso 05_ajuste_dias              HOM=pares FUENTE_HOM=analisis/04b_homogeneidad_pares_laxo.csv DIR_BASE=data/dias SALIDA_AJ=data/dias_ajustados_pares_laxo SUFIJO=_dia_laxo $PY analisis/05_ajuste.py
paso 03_dias                     CRITERIO=anomalias DIR_NOCHES=data/dias MIN_VALIDAS=30 MIN_EXTREMOS=5 SUFIJO=_dia_laxo $PY analisis/03_tendencias.py
paso 03_dias_aj                  CRITERIO=anomalias DIR_NOCHES=data/dias_ajustados_pares_laxo MIN_VALIDAS=30 MIN_EXTREMOS=5 SUFIJO=_dia_laxo_aj $PY analisis/03_tendencias.py
paso 27_reportes_saturados       $PY analisis/27_cambio_sensor.py

echo "== 3. Uso del suelo, urbanización y clima"
paso 13_verano_100               CULTIVO=verano $PY analisis/13_uso_suelo.py
for r in 50 150; do paso "13_verano_$r" CULTIVO=verano RADIO_KM=$r $PY analisis/13_uso_suelo.py; done
for c in soja maiz girasol trigo; do paso "13_$c" CULTIVO=$c $PY analisis/13_uso_suelo.py; done
paso 14_uso_suelo_mensual        $PY analisis/14_uso_suelo_mensual.py
paso 15_lluvia                   $PY analisis/15_lluvia.py
paso 11_enso                     $PY analisis/11_enso.py
paso 11b_enso_sam                $PY analisis/11b_enso_sam.py
paso 28_pdo                      $PY analisis/28_pdo.py
paso 17_indice_salud             $PY analisis/17_indice_salud.py
paso 18_urbano                   $PY analisis/18_urbano.py
paso 24_nulo_espacial            $PY analisis/24_nulo_espacial.py
paso 19_huella_temporal          $PY analisis/19_huella_temporal.py
paso 19b_diferencia_grupos       $PY analisis/19b_diferencia_grupos.py
paso 23_curva_especificacion     $PY analisis/23_multiverso.py
paso 25_mapbiomas                $PY analisis/25_mapbiomas.py
paso 26_mapbiomas_analisis       $PY analisis/26_mapbiomas_analisis.py
paso 26b_intensificacion         $PY analisis/26b_intensificacion.py
paso 29_ciclo_diurno_ond         $PY analisis/29_ciclo_diurno.py
paso 29_ciclo_diurno_ondjfm      MESES=10,11,12,1,2,3 SUFIJO=_ondjfm $PY analisis/29_ciclo_diurno.py
paso 30_viento_nubes             $PY analisis/30_viento_nubes.py
paso 31_extremos                 $PY analisis/31_extremos.py
paso 33_uruguay_paraguay         $PY analisis/33_uy_py.py
paso 34_niebla                   $PY analisis/34_niebla.py
paso 35_ndvi                     $PY analisis/35_ndvi.py
paso 36_ndvi_panel               $PY analisis/36_ndvi_panel.py
paso 37_dipolo                   $PY analisis/37_dipolo.py
paso 40_humedad_suelo            $PY analisis/40_humedad_suelo.py

echo "== 4. Circulación, radiosondeos y ERA5"
paso 07_igra                     $PY analisis/07_igra.py
paso 08_jet_radiosondeo          $PY analisis/08_jet_radiosondeo.py
paso 09_jet_era5                 $PY analisis/09_jet_era5.py
paso 12_descomposicion_ondjfm    $PY analisis/12_descomposicion.py
paso 12_descomposicion_ond       VENTANA=10,11,12 SUFIJO=_ond $PY analisis/12_descomposicion.py
paso 12b_tipos_observados        $PY analisis/12b_descomposicion_igra.py
paso 12c_sensibilidad_tipos      $PY analisis/12c_descomposicion_sensibilidad.py
paso 16_flujo_local              $PY analisis/16_flujo_local.py
paso 20_era5_dependencia         $PY analisis/20_era5_dependencia.py
paso 21_omr_ond                  VENTANA=10,11,12 $PY analisis/21_omr.py
paso 21_omr_ondjfm               VENTANA=10,11,12,1,2,3 $PY analisis/21_omr.py
paso 22_perfil_vertical          COMUN=1 $PY analisis/22_perfil_vertical.py
paso 41_era5_final               $PY analisis/41_era5_final.py
paso 39_controles_revision       $PY analisis/39_revision_controles.py

echo "== 5. Figuras y manuscrito"
paso figuras                     $PY analisis/figuras_paper.py
paso manuscrito_pdf              $PY analisis/borrador_html.py manuscrito
paso suplemento_pdf              $PY analisis/borrador_html.py suplemento
echo "Listo. Resultados en analisis/, figuras en figuras/paper/, PDF en paper/."
