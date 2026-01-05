# %%
pip install numpy pandas matplotlib seaborn pyyaml

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from collections import defaultdict
import numbers
from statistics import mean
import json

# %%
path_to_files = '../nsight_gr_results/'

versiones = ["CS_1st", "CS_2nd", "CS_std", "CU_1st", "CU_2nd"]
moleculas = ["1aga", "1c0o", "2mjq", "8wql"]
rangos = [
    "range1", 
    "range2",
    "range3",
    "range4",
    "range5"
    ]

path_archivos = {}
path_frames = {}

for version in versiones:
    path_archivos[version] = {}
    path_frames[version] = {}
    for molecula in moleculas:
        path_archivos[version][molecula] = {}
        path_frames[version][molecula] = {}
        for rango in rangos:
            path_archivos[version][molecula][rango] = [f"{path_to_files}{version}/{molecula}/YAML/{rango}_1_analysis.yaml",
                                                       f"{path_to_files}{version}/{molecula}/YAML/{rango}_2_analysis.yaml",
                                                       f"{path_to_files}{version}/{molecula}/YAML/{rango}_3_analysis.yaml"]
            path_frames[version][molecula][rango] = f"{path_to_files}{version}/{molecula}/YAML/{rango}_frames.txt"


# %%
def guardar_resultados_json(resultados, nombre_archivo="resultados_analisis.json"):
    """Guarda la lista de diccionarios en un archivo JSON."""
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            # indent=4 hace que el archivo sea legible para humanos
            json.dump(resultados, f, ensure_ascii=False, indent=4)
        print(f"✅ Datos guardados exitosamente en: {nombre_archivo}")
    except Exception as e:
        print(f"❌ Error al guardar JSON: {e}")

def cargar_resultados_json(nombre_archivo="resultados_analisis.json"):
    """Lee el archivo JSON y lo devuelve como lista de Python."""
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        print(f"✅ Datos cargados: {len(datos)} rangos encontrados.")
        return datos
    except Exception as e:
        print(f"❌ Error al cargar JSON: {e}")
        return []

# %%
def cargar_tiempos_frames(path_frames):
    """
    Lee el archivo de tiempos y devuelve una lista de listas.
    Ej: [[70.38, 70.36...], [257.35, 256.90...], ...]
    """
    bloques_tiempos = []
    try:
        with open(path_frames, 'r', encoding='utf-8') as f:
            contenido = f.read().strip()
            
        # Separamos por los guiones que indicaste
        partes = contenido.split('------')
        
        for p in partes:
            if p.strip(): # Evitar bloques vacíos
                # Convertimos cada línea en un número flotante
                tiempos = [float(linea.strip()) for linea in p.strip().splitlines() if linea.strip()]
                bloques_tiempos.append(tiempos)
                
        print(f"Se cargaron {len(bloques_tiempos)} bloques de tiempos de frames.")
        return bloques_tiempos
    except Exception as e:
        print(f"Error cargando tiempos de frames: {e}")
        return []

# %%
def leer_y_analizar_yaml(archivo_path):
    # 1. Leer el archivo YAML
    with open(archivo_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Diccionario para agrupar todas las entradas por su nombre de 'Range'
    # Ejemplo: 'Cleaning Shader': [entrada1, entrada2, ...]
    grupos_rangos = {}
    
    for entrada in data:
        nombre_rango = entrada.get('Range', 'Desconocido')
        if nombre_rango not in grupos_rangos:
            grupos_rangos[nombre_rango] = []
        grupos_rangos[nombre_rango].append(entrada)

    resultado_final = []

    # 2. Procesar cada grupo
    for nombre, lista_entradas in grupos_rangos.items():
        # Usamos la primera entrada como "plantilla" para la estructura de Reglas/Métricas
        estructura_base = lista_entradas[0]
        
        # Objeto para guardar el resumen de este Rango
        rango_resumen = {
            'Range': nombre,
            'Count': len(lista_entradas), # Cuántas veces aparece este rango
            'RelativeFrameDuration_Avg': 0,
            'Rules': []
        }

        # Promediar RelativeFrameDuration (si existe)
        duraciones = [e.get('RelativeFrameDuration', 0) for e in lista_entradas if 'RelativeFrameDuration' in e]
        if duraciones:
            rango_resumen['RelativeFrameDuration_Avg'] = mean(duraciones)

        # 3. Agrupar y procesar Rules (Reglas)
        # Suponemos que el orden y nombres de las reglas son consistentes
        if 'Rules' in estructura_base and estructura_base['Rules']:
            for i, regla_base in enumerate(estructura_base['Rules']):
                nombre_regla = regla_base.get('Name')
                
                regla_resumen = {
                    'Name': nombre_regla,
                    'Category': regla_base.get('Category'),
                    'Metrics': []
                }
                
                # Recolectar textos únicos para Explanation/Suggestion
                explicaciones = set()
                sugerencias = set()
                
                # Buscar esta misma regla en todas las entradas del grupo
                reglas_coincidentes = []
                for entrada in lista_entradas:
                    # Buscamos la regla por nombre en la lista de reglas de la entrada actual
                    reglas_entrada = entrada.get('Rules', [])
                    match = next((r for r in reglas_entrada if r.get('Name') == nombre_regla), None)
                    if match:
                        reglas_coincidentes.append(match)
                        if match.get('Explanation'): explicaciones.add(match['Explanation'])
                        if match.get('Suggestion'): sugerencias.add(match['Suggestion'])
                
                regla_resumen['Explanation_Accumulated'] = list(explicaciones)
                regla_resumen['Suggestion_Accumulated'] = list(sugerencias)

                # 4. Procesar Metrics (Métricas) dentro de la regla
                if 'Metrics' in regla_base and regla_base['Metrics']:
                    for metrica_base in regla_base['Metrics']:
                        id_metrica = metrica_base.get('Id')
                        
                        # Recolectar valores de esta métrica en todas las instancias
                        valores_numericos = []
                        
                        for r_coincidente in reglas_coincidentes:
                            metrics_list = r_coincidente.get('Metrics', [])
                            m_match = next((m for m in metrics_list if m.get('Id') == id_metrica), None)
                            
                            if m_match and 'Value' in m_match:
                                val = m_match['Value']
                                if isinstance(val, (int, float)):
                                    valores_numericos.append(val)
                        
                        metrica_resumen = {
                            'Name': metrica_base.get('Name'),
                            'Id': id_metrica,
                            'Description': metrica_base.get('Description'),
                            'Value_Avg': mean(valores_numericos) if valores_numericos else 0,
                            'Value_Min': min(valores_numericos) if valores_numericos else 0,
                            'Value_Max': max(valores_numericos) if valores_numericos else 0
                        }
                        regla_resumen['Metrics'].append(metrica_resumen)
                
                rango_resumen['Rules'].append(regla_resumen)

        resultado_final.append(rango_resumen)

    return resultado_final

# %%
def analizar_con_tiempos_reales(lista_paths_yaml, path_frames, RANGOS_POR_FRAME = 5):
    # 1. Cargar los tiempos de referencia
    # Asumimos que el orden de los bloques en .txt coincide con el orden de lista_paths_yaml
    tiempos_por_archivo = cargar_tiempos_frames(path_frames)
    
    if len(tiempos_por_archivo) != len(lista_paths_yaml):
        print("ADVERTENCIA: La cantidad de archivos YAML no coincide con los bloques de tiempos en el .txt")

    data_total = []
      # Dato que me diste: cada frame tiene 5 rangos

    print("\nProcesando y sincronizando archivos...")
    
    # 2. Iterar archivos YAML sincronizados con sus tiempos
    for i, archivo_path in enumerate(lista_paths_yaml):
        try:
            # Seleccionamos la lista de tiempos correspondiente a este archivo
            tiempos_frames_actuales = tiempos_por_archivo[i] if i < len(tiempos_por_archivo) else []
            
            with open(archivo_path, 'r', encoding='utf-8') as f:
                contenido = yaml.safe_load(f)
                
                if isinstance(contenido, list):
                    # --- SINCRONIZACIÓN DE TIEMPOS ---
                    for index_tupla, entrada in enumerate(contenido):
                        # Calculamos a qué número de frame pertenece esta tupla
                        # Ej: tupla 0-4 -> frame 0; tupla 5-9 -> frame 1
                        numero_frame = index_tupla // RANGOS_POR_FRAME
                        
                        if numero_frame < len(tiempos_frames_actuales):
                            tiempo_total_este_frame = tiempos_frames_actuales[numero_frame]
                            rel_duration = entrada.get('RelativeFrameDuration', 0)
                            
                            # CÁLCULO DEL TIEMPO REAL (ms)
                            # Asumiendo que RelativeFrameDuration es 0.0-1.0 (ej: 0.5 = 50%)
                            # Si en tus datos es 0-100, divide por 100.
                            tiempo_real_ms = tiempo_total_este_frame * rel_duration
                            
                            # Inyectamos este valor en la entrada para usarlo después
                            entrada['Real_Duration_Calculated_ms'] = tiempo_real_ms
                            entrada['Frame_Total_Time_Ref'] = tiempo_total_este_frame
                        else:
                            # Si hay más tuplas que tiempos registrados
                            entrada['Real_Duration_Calculated_ms'] = 0

                    data_total.extend(contenido)
                    print(f" -> {archivo_path}: Procesado correctamente.")
                else:
                    print(f" -> {archivo_path}: Formato incorrecto.")
                    
        except Exception as e:
            print(f" -> Error en {archivo_path}: {e}")

    # 3. Agrupar (Lógica similar a la anterior, agregando el promedio de tiempo real)
    grupos_rangos = {}
    for entrada in data_total:
        nombre = entrada.get('Range', 'Desconocido')
        if nombre not in grupos_rangos: grupos_rangos[nombre] = []
        grupos_rangos[nombre].append(entrada)

    resultado_final = []

    for nombre, lista_entradas in grupos_rangos.items():
        estructura_base = lista_entradas[0]
        
        # Recolectar métricas básicas del rango
        lista_real_ms = [e['Real_Duration_Calculated_ms'] for e in lista_entradas if 'Real_Duration_Calculated_ms' in e]
        lista_rel = [e.get('RelativeFrameDuration', 0) for e in lista_entradas]
        
        rango_resumen = {
            'Range': nombre,
            'Total_Muestras': len(lista_entradas),
            'RelativeFrameDuration_Avg': mean(lista_rel) if lista_rel else 0,
            # --- NUEVA MÉTRICA IMPORTANTE ---
            'Real_Duration_ms_Avg': mean(lista_real_ms) if lista_real_ms else 0,
            'Rules': []
        }

        # Procesar Reglas (Lógica de agrupación de texto y números)
        if 'Rules' in estructura_base and estructura_base['Rules']:
            for regla_base in estructura_base['Rules']:
                nombre_regla = regla_base.get('Name')
                regla_resumen = {
                    'Name': nombre_regla, 
                    'Category': regla_base.get('Category'),
                    'Metrics': []
                }
                
                explicaciones = set()
                sugerencias = set()
                frame_gains = []
                speedups = []
                
                reglas_coincidentes = []
                
                for entrada in lista_entradas:
                    match = next((r for r in entrada.get('Rules', []) if r.get('Name') == nombre_regla), None)
                    if match:
                        reglas_coincidentes.append(match)
                        if match.get('Explanation'): explicaciones.add(match['Explanation'])
                        if match.get('Suggestion'): sugerencias.add(match['Suggestion'])
                        if match.get('FrameGain'): frame_gains.append(match['FrameGain'])
                        if match.get('RangeSpeedupFactor'): speedups.append(match['RangeSpeedupFactor'])

                regla_resumen['Explanation_Accumulated'] = list(explicaciones)
                regla_resumen['Suggestion_Accumulated'] = list(sugerencias)
                regla_resumen['FrameGain_Avg'] = mean(frame_gains) if frame_gains else 0
                regla_resumen['RangeSpeedupFactor_Avg'] = mean(speedups) if speedups else 0

                # Procesar Métricas internas
                if 'Metrics' in regla_base and regla_base['Metrics']:
                    for metrica_base in regla_base['Metrics']:
                        id_metrica = metrica_base.get('Id')
                        valores = []
                        for rc in reglas_coincidentes:
                            m_match = next((m for m in rc.get('Metrics', []) if m.get('Id') == id_metrica), None)
                            if m_match and isinstance(m_match.get('Value'), (int, float)):
                                valores.append(m_match['Value'])
                        
                        metrica_resumen = {
                            'Name': metrica_base.get('Name'),
                            'Id': id_metrica,
                            'Value_Avg': mean(valores) if valores else 0,
                            'Value_Min': min(valores) if valores else 0,
                            'Value_Max': max(valores) if valores else 0
                        }
                        regla_resumen['Metrics'].append(metrica_resumen)
                
                rango_resumen['Rules'].append(regla_resumen)

        resultado_final.append(rango_resumen)

    return resultado_final

# %%
cs_1st_1aga_ranges = []
for i in range(5):
    cs_1st_1aga_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_1st"]["1aga"][f"range{i+1}"], path_frames["CS_1st"]["1aga"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_1st_1aga_ranges[0]
rearrange["range2"] = cs_1st_1aga_ranges[1]
rearrange["range3"] = cs_1st_1aga_ranges[2]
rearrange["range4"] = cs_1st_1aga_ranges[3]
rearrange["range5"] = cs_1st_1aga_ranges[4]

guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_1st_1aga_ranges.json")

# %%
cs_1st_1c0o_ranges = []
for i in range(5):
    cs_1st_1c0o_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_1st"]["1c0o"][f"range{i+1}"], path_frames["CS_1st"]["1c0o"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_1st_1c0o_ranges[0]
rearrange["range2"] = cs_1st_1c0o_ranges[1]
rearrange["range3"] = cs_1st_1c0o_ranges[2]
rearrange["range4"] = cs_1st_1c0o_ranges[3]
rearrange["range5"] = cs_1st_1c0o_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_1st_1c0o_ranges.json")

# %%
cs_1st_2mjq_ranges = []
for i in range(5):
    cs_1st_2mjq_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_1st"]["2mjq"][f"range{i+1}"], path_frames["CS_1st"]["2mjq"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_1st_2mjq_ranges[0]
rearrange["range2"] = cs_1st_2mjq_ranges[1]
rearrange["range3"] = cs_1st_2mjq_ranges[2]
rearrange["range4"] = cs_1st_2mjq_ranges[3]
rearrange["range5"] = cs_1st_2mjq_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_1st_2mjq_ranges.json")

# %%
cs_1st_8wql_ranges = []
for i in range(5):
    cs_1st_8wql_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_1st"]["8wql"][f"range{i+1}"], path_frames["CS_1st"]["8wql"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_1st_8wql_ranges[0]
rearrange["range2"] = cs_1st_8wql_ranges[1]
rearrange["range3"] = cs_1st_8wql_ranges[2]
rearrange["range4"] = cs_1st_8wql_ranges[3]
rearrange["range5"] = cs_1st_8wql_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_1st_8wql_ranges.json")

# %%
def plotear_resultados_por_rango(data_ranges, titulos_eje_x):
    """
    data_ranges: Lista de listas con los diccionarios procesados por cada paso.
    titulos_eje_x: Lista de strings para el eje X (ej: ['range1', 'range2'...])
    """
    
    # 1. Identificar todos los nombres únicos de "Range" (Shaders) presentes en los datos
    nombres_unicos_rangos = set()
    for paso in data_ranges:
        for item in paso:
            nombres_unicos_rangos.add(item['Range'])
    
    # Ordenamos para consistencia
    nombres_unicos_rangos = sorted(list(nombres_unicos_rangos))

    print(f"Se encontraron {len(nombres_unicos_rangos)} tipos de rangos para graficar: {nombres_unicos_rangos}")

    # 2. Generar un plot por cada Nombre de Rango
    for nombre_rango in nombres_unicos_rangos:
        valores_y = []
        
        # Recorremos los 5 pasos (range1, range2, etc.)
        for paso_index, paso_data in enumerate(data_ranges):
            # Buscamos si este 'nombre_rango' existe en este paso
            dato = next((item for item in paso_data if item['Range'] == nombre_rango), None)
            
            if dato:
                valores_y.append(dato['Real_Duration_ms_Avg'])
            else:
                valores_y.append(0) # Si no aparece en este paso, asumimos 0

        # Crear la figura
        plt.figure(figsize=(10, 6))
        
        # Gráfico de barras (o plot si prefieres líneas)
        plt.bar(titulos_eje_x, valores_y, color='skyblue', edgecolor='navy')
        # Opcional: Línea de tendencia
        plt.plot(titulos_eje_x, valores_y, color='red', marker='o', linestyle='--', label='Tendencia')

        plt.title(f'Evolución de Duración: {nombre_rango}')
        plt.xlabel('Datasets')
        plt.ylabel('Relative Frame Duration Avg')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Mostrar el valor encima de las barras
        for i, v in enumerate(valores_y):
            plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

# %%
etiquetas_x = ["range1", "range2", "range3", "range4", "range5"]

# Asegúrate de ejecutar esto después de haber llenado 'cs_1st_1aga_ranges'
if 'cs_1st_1aga_ranges' in locals() and cs_1st_1aga_ranges:
    plotear_resultados_por_rango(cs_1st_1aga_ranges, etiquetas_x)
else:
    print("La variable 'cs_1st_1aga_ranges' no está definida o está vacía.")

# %%
if 'cs_1st_1c0o_ranges' in locals() and cs_1st_1c0o_ranges:
    plotear_resultados_por_rango(cs_1st_1c0o_ranges, etiquetas_x)
else:
    print("La variable 'cs_1st_1c0o_ranges' no está definida o está vacía.")

# %%
if 'cs_1st_2mjq_ranges' in locals() and cs_1st_2mjq_ranges:
    plotear_resultados_por_rango(cs_1st_2mjq_ranges, etiquetas_x)
else:
    print("La variable 'cs_1st_2mjq_ranges' no está definida o está vacía.")

# %%
if 'cs_1st_8wql_ranges' in locals() and cs_1st_8wql_ranges:
    plotear_resultados_por_rango(cs_1st_8wql_ranges, etiquetas_x)
else:
    print("La variable 'cs_1st_8wql_ranges' no está definida o está vacía.")

# %%
cs_2nd_1aga_ranges = []
for i in range(5):
    cs_2nd_1aga_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_2nd"]["1aga"][f"range{i+1}"], path_frames["CS_2nd"]["1aga"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_2nd_1aga_ranges[0]
rearrange["range2"] = cs_2nd_1aga_ranges[1]
rearrange["range3"] = cs_2nd_1aga_ranges[2]
rearrange["range4"] = cs_2nd_1aga_ranges[3]
rearrange["range5"] = cs_2nd_1aga_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_2nd_1aga_ranges.json")

# %%
cs_2nd_1c0o_ranges = []
for i in range(5):
    cs_2nd_1c0o_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_2nd"]["1c0o"][f"range{i+1}"], path_frames["CS_2nd"]["1c0o"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_2nd_1c0o_ranges[0]
rearrange["range2"] = cs_2nd_1c0o_ranges[1]
rearrange["range3"] = cs_2nd_1c0o_ranges[2]
rearrange["range4"] = cs_2nd_1c0o_ranges[3]
rearrange["range5"] = cs_2nd_1c0o_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_2nd_1c0o_ranges.json")

# %%
cs_2nd_2mjq_ranges = []
for i in range(5):
    cs_2nd_2mjq_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_2nd"]["2mjq"][f"range{i+1}"], path_frames["CS_2nd"]["2mjq"][f"range{i+1}"]))

# %%
rearrange = {}
rearrange["range1"] = cs_2nd_2mjq_ranges[0]
rearrange["range2"] = cs_2nd_2mjq_ranges[1]
rearrange["range3"] = cs_2nd_2mjq_ranges[2]
rearrange["range4"] = cs_2nd_2mjq_ranges[3]
rearrange["range5"] = cs_2nd_2mjq_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_2nd_2mjq_ranges.json")

# %%
if 'cs_2nd_1aga_ranges' in locals() and cs_2nd_1aga_ranges:
    plotear_resultados_por_rango(cs_2nd_1aga_ranges, etiquetas_x)
else:
    print("La variable 'cs_2nd_1aga_ranges' no está definida o está vacía.")

# %%
if 'cs_2nd_1c0o_ranges' in locals() and cs_2nd_1c0o_ranges:
    plotear_resultados_por_rango(cs_2nd_1c0o_ranges, etiquetas_x)
else:
    print("La variable 'cs_2nd_1c0o_ranges' no está definida o está vacía.")

# %%
if 'cs_2nd_2mjq_ranges' in locals() and cs_2nd_2mjq_ranges:
    plotear_resultados_por_rango(cs_2nd_2mjq_ranges, etiquetas_x)
else:
    print("La variable 'cs_2nd_2mjq_ranges' no está definida o está vacía.")

# %%
cs_std_1aga_ranges = []
for i in range(5):
    cs_std_1aga_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_std"]["1aga"][f"range{i+1}"], path_frames["CS_std"]["1aga"][f"range{i+1}"]))

rearrange = {}
rearrange["range1"] = cs_std_1aga_ranges[0]
rearrange["range2"] = cs_std_1aga_ranges[1]
rearrange["range3"] = cs_std_1aga_ranges[2]
rearrange["range4"] = cs_std_1aga_ranges[3]
rearrange["range5"] = cs_std_1aga_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_std_1aga_ranges.json")

# %%
cs_std_1c0o_ranges = []
for i in range(5):
    cs_std_1c0o_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_std"]["1c0o"][f"range{i+1}"], path_frames["CS_std"]["1c0o"][f"range{i+1}"]))

rearrange = {}
rearrange["range1"] = cs_std_1c0o_ranges[0]
rearrange["range2"] = cs_std_1c0o_ranges[1]
rearrange["range3"] = cs_std_1c0o_ranges[2]
rearrange["range4"] = cs_std_1c0o_ranges[3]
rearrange["range5"] = cs_std_1c0o_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_std_1c0o_ranges.json")

# %%
cs_std_2mjq_ranges = []
for i in range(5):
    cs_std_2mjq_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_std"]["2mjq"][f"range{i+1}"], path_frames["CS_std"]["2mjq"][f"range{i+1}"]))

rearrange = {}
rearrange["range1"] = cs_std_2mjq_ranges[0]
rearrange["range2"] = cs_std_2mjq_ranges[1]
rearrange["range3"] = cs_std_2mjq_ranges[2]
rearrange["range4"] = cs_std_2mjq_ranges[3]
rearrange["range5"] = cs_std_2mjq_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_std_2mjq_ranges.json")

# %%
cs_std_8wql_ranges = []
for i in range(5):
    cs_std_8wql_ranges.append(analizar_con_tiempos_reales(path_archivos["CS_std"]["8wql"][f"range{i+1}"], path_frames["CS_std"]["8wql"][f"range{i+1}"]))

rearrange = {}
rearrange["range1"] = cs_std_8wql_ranges[0]
rearrange["range2"] = cs_std_8wql_ranges[1]
rearrange["range3"] = cs_std_8wql_ranges[2]
rearrange["range4"] = cs_std_8wql_ranges[3]
rearrange["range5"] = cs_std_8wql_ranges[4]
guardar_resultados_json(rearrange, nombre_archivo="resultados_cs_std_8wql_ranges.json")

# %%
if 'cs_std_1aga_ranges' in locals() and cs_std_1aga_ranges:
    plotear_resultados_por_rango(cs_std_1aga_ranges, etiquetas_x)
else:
    print("La variable 'cs_std_1aga_ranges' no está definida o está vacía.")

# %%
if 'cs_std_1c0o_ranges' in locals() and cs_std_1c0o_ranges:
    plotear_resultados_por_rango(cs_std_1c0o_ranges, etiquetas_x)
else:
    print("La variable 'cs_std_1c0o_ranges' no está definida o está vacía.")

# %%
if 'cs_std_2mjq_ranges' in locals() and cs_std_2mjq_ranges:
    plotear_resultados_por_rango(cs_std_2mjq_ranges, etiquetas_x)
else:
    print("La variable 'cs_std_2mjq_ranges' no está definida o está vacía.")

# %%
if 'cs_std_8wql_ranges' in locals() and cs_std_8wql_ranges:
    plotear_resultados_por_rango(cs_std_8wql_ranges, etiquetas_x)
else:
    print("La variable 'cs_std_8wql_ranges' no está definida o está vacía.")

# %%
# --- EXTENSIÓN: ANÁLISIS AVANZADO CON PANDAS ---

def consolidar_datos_en_dataframe(diccionario_datos_cargados):
    """
    Transforma tu estructura de listas/diccionarios en un DataFrame único.
    diccionario_datos_cargados: Un dict donde la clave es el nombre del experimento
    (ej: "CS_1st_1aga") y el valor es la lista de rangos [range1, range2...]
    """
    filas = []
    
    for nombre_experimento, lista_rangos in diccionario_datos_cargados.items():
        # Parseamos el nombre (ej: CS_1st_1aga -> Version: CS_1st, Molecula: 1aga)
        # Ajusta esto según cómo nombres tus claves
        partes = nombre_experimento.split('_')
        version = f"{partes[0]}_{partes[1]}" if len(partes) >= 2 else "Desconocida"
        molecula = partes[2] if len(partes) >= 3 else "Desconocida"

        for i, datos_rango in enumerate(lista_rangos):
            nombre_dataset = f"range{i+1}"
            
            # datos_rango es una lista de diccionarios (uno por cada Shader/Range encontrado)
            for item in datos_rango:
                filas.append({
                    'Experimento': nombre_experimento,
                    'Version': version,
                    'Molecula': molecula,
                    'Dataset': nombre_dataset,
                    'Shader_Name': item.get('Range', 'Unknown'),
                    'Duration_ms': item.get('Real_Duration_ms_Avg', 0),
                    'Rel_Duration': item.get('RelativeFrameDuration_Avg', 0),
                    'Muestras': item.get('Total_Muestras', 0)
                })

    return pd.DataFrame(filas)


def consolidar_metricas_detalladas(diccionario_datos_cargados):
    """
    Recorre la estructura profunda (Rules -> Metrics) y extrae cada métrica individual.
    Devuelve un DataFrame listo para graficar con Seaborn.
    """
    filas = []
    
    for nombre_experimento, data_rangos in diccionario_datos_cargados.items():
        # Parseo del nombre (ej: CS_1st_1aga)
        partes = nombre_experimento.split('_')
        # Asumiendo estructura "Version_Subversion_Molecula" o "Version_Molecula"
        # Ajusta indices si tu naming convention es distinto
        if len(partes) >= 3:
            version = f"{partes[0]}_{partes[1]}" # Ej: CS_1st
            molecula = partes[2]                 # Ej: 1aga
        else:
            version = partes[0]
            molecula = "Unknown"

        # Manejo flexible: si 'data_rangos' es lista (variables en memoria) o dict (cargado de JSON)
        if isinstance(data_rangos, list):
            iterador = enumerate(data_rangos) # (0, data_range1), (1, data_range2)...
        elif isinstance(data_rangos, dict):
            iterador = enumerate(data_rangos.values()) # Asumimos orden correcto o keys ordenadas
        else:
            continue

        for i, lista_shaders in iterador:
            nombre_range = f"range{i+1}"
            
            for shader_data in lista_shaders:
                shader_name = shader_data.get('Range', 'Unknown')
                
                # Iterar sobre las Reglas (Categorías de métricas)
                for regla in shader_data.get('Rules', []):
                    regla_nombre = regla.get('Name')
                    
                    # Iterar sobre las Métricas individuales
                    for metrica in regla.get('Metrics', []):
                        filas.append({
                            'Experimento': nombre_experimento,
                            'Version': version,
                            'Molecula': molecula,
                            'Dataset': nombre_range, # range1, range2...
                            'Shader': shader_name,
                            'Rule_Name': regla_nombre,       # Ej: GPU Engines Active
                            'Metric_Name': metrica.get('Name'), # Ej: GR Cycles Active [%]
                            'Metric_Id': metrica.get('Id'),
                            'Value_Avg': metrica.get('Value_Avg', 0),
                            'Value_Min': metrica.get('Value_Min', 0),
                            'Value_Max': metrica.get('Value_Max', 0)
                        })

    return pd.DataFrame(filas)


# %%
# --- USO ---
# 1. Agrupamos tus variables actuales en un diccionario
# (Asegúrate de que estas variables existen en tu entorno)
datos_para_analisis = {
    "CS_1st_1aga": cs_1st_1aga_ranges,
    "CS_1st_1c0o": cs_1st_1c0o_ranges,
    "CS_1st_2mjq": cs_1st_2mjq_ranges,
    "CS_1st_8wql": cs_1st_8wql_ranges
}

# 2. Creamos el DataFrame
df_cs_1st_master = consolidar_datos_en_dataframe(datos_para_analisis)

# Verificamos
print(f"✅ DataFrame creado con {len(df_cs_1st_master)} filas.")
display(df_cs_1st_master.head())

# %%
# Agrupar por Shader y calcular tiempo promedio y total
top_shaders = df_cs_1st_master.groupby('Shader_Name')['Duration_ms'].agg(['mean', 'sum', 'count']).sort_values(by='mean', ascending=False)

print("\nTop 5 Shaders más costosos (Promedio por llamada):")
display(top_shaders.head(5))

# Gráfico
plt.figure(figsize=(12, 6))
sns.barplot(x=top_shaders.head(10).index, y=top_shaders.head(10)['mean'], palette='viridis')
plt.title('Top 10 Shaders por Duración Promedio (ms)')
plt.ylabel('Duración Promedio (ms)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

# %%
# Pivotar datos: Filas=Shaders, Columnas=Molecula, Valor=Duración
heatmap_data = df_cs_1st_master.pivot_table(index='Shader_Name', columns='Molecula', values='Duration_ms', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=.5)
plt.title('Heatmap: Costo de Shaders (ms) por Molécula')
plt.show()

# %%
plt.figure(figsize=(14, 6))
# Filtramos solo los shaders principales para no saturar el gráfico
top_n_shaders = df_cs_1st_master.groupby('Shader_Name')['Duration_ms'].sum().nlargest(5).index
df_filtered = df_cs_1st_master[df_cs_1st_master['Shader_Name'].isin(top_n_shaders)]

sns.boxplot(data=df_filtered, x='Shader_Name', y='Duration_ms', hue='Molecula')
plt.title('Distribución de Tiempos por Shader y Molécula (Estabilidad)')
plt.xticks(rotation=45)
plt.ylabel('Duración (ms)')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# --- USO ---
# 1. Agrupamos tus variables actuales en un diccionario
# (Asegúrate de que estas variables existen en tu entorno)
datos_para_analisis = {
    "CS_2nd_1aga": cs_2nd_1aga_ranges,
    "CS_2nd_1c0o": cs_2nd_1c0o_ranges,
    "CS_2nd_2mjq": cs_2nd_2mjq_ranges
}

# 2. Creamos el DataFrame
df_cs_2nd_master = consolidar_datos_en_dataframe(datos_para_analisis)

# Verificamos
print(f"✅ DataFrame creado con {len(df_cs_2nd_master)} filas.")
display(df_cs_2nd_master.head())

# %%
# --- USO ---
# 1. Agrupamos tus variables actuales en un diccionario
# (Asegúrate de que estas variables existen en tu entorno)
datos_para_analisis = {
    "CS_std_1aga": cs_std_1aga_ranges,
    "CS_std_1c0o": cs_std_1c0o_ranges,
    "CS_std_2mjq": cs_std_2mjq_ranges,
    "CS_std_8wql": cs_std_8wql_ranges
}

# 2. Creamos el DataFrame
df_cs_std_master = consolidar_datos_en_dataframe(datos_para_analisis)

# Verificamos
print(f"✅ DataFrame creado con {len(df_cs_std_master)} filas.")
display(df_cs_std_master.head())

# %%
todos_los_datos = {
    "CS_1st_1aga": cs_1st_1aga_ranges,
    "CS_1st_1c0o": cs_1st_1c0o_ranges,
    "CS_1st_2mjq": cs_1st_2mjq_ranges,
    "CS_1st_8wql": cs_1st_8wql_ranges,
    "CS_2nd_1aga": cs_2nd_1aga_ranges,
    "CS_2nd_1c0o": cs_2nd_1c0o_ranges,
    "CS_2nd_2mjq": cs_2nd_2mjq_ranges,
    "CS_std_1aga": cs_std_1aga_ranges,
    "CS_std_1c0o": cs_std_1c0o_ranges,
    "CS_std_2mjq": cs_std_2mjq_ranges,
    "CS_std_8wql": cs_std_8wql_ranges
}

df_detailed_metrics = consolidar_metricas_detalladas(todos_los_datos)

print(f"DataFrame generado con {len(df_detailed_metrics)} registros de métricas.")
display(df_detailed_metrics.head())
display(df_detailed_metrics["Rule_Name"].unique())

# %%
def graficar_metrica_comparativa(df, shader_target, metric_name_filter, by_molecule=False):
    """
    df: El DataFrame generado con consolidar_metricas_detalladas.
    shader_target: Nombre exacto del shader (ej: 'Cleaning Shader').
    metric_name_filter: Parte del nombre de la métrica (ej: 'GR Cycles Active').
    by_molecule: Si True, separa gráficos por molécula. Si False, compara versiones agrupando moléculas.
    """
    # 1. Filtrar datos
    # Buscamos filas del shader deseado y cuya métrica contenga el texto filtro
    mask = (df['Shader'] == shader_target) & \
           (df['Metric_Name'].str.contains(metric_name_filter, case=False, na=False))
    
    data_plot = df[mask].copy()
    
    if data_plot.empty:
        print(f"⚠️ No se encontraron datos para Shader='{shader_target}' y Metrica~='{metric_name_filter}'")
        return

    # Obtenemos el nombre completo de la métrica para el título (usamos el primero que aparezca)
    full_metric_name = data_plot['Metric_Name'].iloc[0]

    # 2. Configurar gráfico
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Si queremos ver diferencias entre moléculas, usamos 'col' de catplot, si no, hue
    if by_molecule:
        g = sns.catplot(
            data=data_plot, 
            x="Dataset", 
            y="Value_Avg", 
            hue="Version", 
            col="Molecula",
            kind="bar", 
            height=5, 
            aspect=1.2,
            errorbar=None # Quitar si quieres ver varianza (si tienes repeticiones)
        )
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f"{full_metric_name}\nEn shader: {shader_target}", fontsize=16)
    else:
        # Comparación directa de versiones (promediando moléculas si hay varias)
        sns.barplot(
            data=data_plot, 
            x="Dataset", 
            y="Value_Avg", 
            hue="Version"
        )
        plt.title(f"{full_metric_name}\nEn shader: {shader_target}", fontsize=15)
        plt.ylabel(full_metric_name)
        plt.xlabel("Rango")
        plt.legend(title="Versión")
        plt.show()


def filtrar_moleculas_insuficientes(df, min_rangos=4):
    """
    Filtra el DataFrame para conservar solo las moléculas que tengan datos
    en al menos 'min_rangos' datasets diferentes para una métrica dada.
    """
    # Calculamos cuántos datasets únicos tiene cada combinación de (Versión, Shader, Métrica, Molécula)
    # Usamos 'transform' para asignar ese conteo a cada fila original
    conteo_rangos = df.groupby(['Version', 'Shader', 'Rule_Name', 'Metric_Name', 'Molecula'])['Dataset'].transform('nunique')
    
    # Filtramos las filas que no cumplen el requisito
    df_filtrado = df[conteo_rangos >= min_rangos].copy()
    
    return df_filtrado

def graficar_regla_por_columnas(df, version_target, shader_target, rule_name='L1TEX L2 Hit Rates', min_rangos=4):
    """
    Grafica las métricas de una regla, aplicando dos filtros de calidad:
    1. Elimina métricas que son 0 en todos los casos.
    2. Elimina moléculas que tienen menos de 'min_rangos' puntos de datos.
    """
    
    # --- PASO 1: Selección inicial de datos ---
    mask = (df['Version'] == version_target) & \
           (df['Shader'] == shader_target) & \
           (df['Rule_Name'] == rule_name)
    
    data_plot = df[mask].copy()
    
    if data_plot.empty:
        print(f"⚠️ No hay datos iniciales para: {rule_name}")
        return

    # --- PASO 2: Filtro de consistencia (Tu nuevo requerimiento) ---
    # Eliminamos moléculas que aparecen en pocos rangos (ej: solo en range1)
    data_plot = filtrar_moleculas_insuficientes(data_plot, min_rangos=min_rangos)
    
    if data_plot.empty:
        print(f"⚠️ Se filtraron todos los datos. Ninguna molécula tiene al menos {min_rangos} rangos.")
        return

    # --- PASO 3: Filtro de relevancia (Eliminar métricas vacías/cero) ---
    # Calculamos el máximo valor absoluto por métrica (considerando solo la data que sobrevivió al paso 2)
    metric_max_values = data_plot.groupby('Metric_Name')['Value_Avg'].apply(lambda x: x.abs().max())
    metrics_to_keep = metric_max_values[metric_max_values > 1e-9].index.tolist()
    data_plot = data_plot[data_plot['Metric_Name'].isin(metrics_to_keep)].copy()
    
    if data_plot.empty:
        print(f"ℹ️ Todas las métricas restantes son cero. No se grafica nada.")
        return

    # --- PASO 4: Preparación para graficar ---
    # Ordenar datasets numéricamente para el eje X
    data_plot['Dataset_Num'] = data_plot['Dataset'].str.extract('(\d+)').astype(int)
    data_plot = data_plot.sort_values('Dataset_Num')

    print(f"Graficando {len(metrics_to_keep)} métricas. (Se aplicó filtro: Mínimo {min_rangos} rangos por molécula)")

    # --- PASO 5: Graficado ---
    g = sns.catplot(
        data=data_plot,
        x="Dataset",
        y="Value_Avg",
        hue="Molecula",       
        col="Metric_Name",    
        col_wrap=3,           # Máximo 3 columnas
        kind="point",         # Líneas con puntos
        sharey=False,         # Escalas independientes
        height=8,
        aspect=1.2,
        palette="viridis",
        markers="o",
        linestyles="-"
    )
    
    g.fig.subplots_adjust(top=0.9) 
    g.fig.suptitle(f"Análisis: '{rule_name}' ({version_target})\nShader {shader_target}", fontsize=15)
    
    g.set_axis_labels("Rango", "Valor")
    g.set_titles("{col_name}") 
    
    plt.show()



# %%
graficar_metrica_comparativa(df_detailed_metrics, "Cleaning Shader", "L1TEX Hit Rate")

graficar_metrica_comparativa(df_detailed_metrics, "Cleaning Shader", "VRAM Throughput")


# %%
rules_to_analyze = df_detailed_metrics["Rule_Name"].unique()

# Estabilidad: Si las líneas de las distintas moléculas (1aga, 1c0o, etc.) están muy juntas, 
# significa que el rendimiento de tu shader es estable independientemente de la geometría 
# específica de la molécula.

# Tendencia: Si al aumentar el rango (ir hacia la derecha en el eje X) la línea baja 
# drásticamente, significa que al alejarse la cámara, el patrón de acceso a memoria 
# se vuelve menos eficiente (cache misses).

# Diferencias de Escala: Gracias a sharey=False, podrás ver claramente variaciones pequeñas 
# en el L1 Hit Rate (que suele estar cerca de 80-90%) sin que se vea aplastado por otra métrica 
# que tenga una escala diferente.
for regla in rules_to_analyze:
    print(f"\n--- Graficando regla: {regla} ---\n")
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_1st", 
        shader_target="Cleaning Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_1st", 
        shader_target="Sphere Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_1st", 
        shader_target="Cylinder Shader", 
        rule_name=regla
    )



# %%
for regla in rules_to_analyze:
    print(f"\n--- Graficando regla: {regla} ---\n")
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_2nd", 
        shader_target="Sphere Bbox Extraction Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_2nd", 
        shader_target="Sphere Bbox Intersection Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_2nd", 
        shader_target="Cylinder Bbox Extraction Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_2nd", 
        shader_target="Cylinder Bbox Intersection Shader", 
        rule_name=regla
    )
    

# %%
for regla in rules_to_analyze:
    print(f"\n--- Graficando regla: {regla} ---\n")
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_std", 
        shader_target="Sphere Culling Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_std", 
        shader_target="Draw Spheres", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_std", 
        shader_target="Cylinder Culling Shader", 
        rule_name=regla
    )
    graficar_regla_por_columnas(
        df_detailed_metrics, 
        version_target="CS_std", 
        shader_target="Draw Cylinders", 
        rule_name=regla
    )
    


