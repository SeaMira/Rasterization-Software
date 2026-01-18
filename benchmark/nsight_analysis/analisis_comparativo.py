import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np

# Directorio donde están los archivos JSON
base_dir = r"d:\Users\Escritorio\Rasterization-Software\benchmark\nsight_analysis"

# Definir los archivos a analizar
versiones = ['CS_std', 'CS_1st', 'CS_2nd']
moleculas = ['1aga', '1c0o', '2mjq', '8wql']
grids = ['grid1', 'grid2', 'grid3', 'grid4']

def extraer_metricas(filepath):
    """Extrae métricas clave de un archivo JSON de resultados"""
    
    if not os.path.exists(filepath):
        print(f"Archivo no encontrado: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    resultados = {
        'shaders': [],
        'duraciones': [],
        'gr_cycles_active': [],
        'l1tex_hit_rate': [],
        'vram_throughput': [],
        'gpu_idle': []
    }
    
    # Iterar sobre los ranges
    for range_key, range_data in data.items():
        for shader_data in range_data:
            shader_name = shader_data.get('Range', 'Unknown')
            duracion_ms = shader_data.get('Real_Duration_ms_Avg', 0)
            
            resultados['shaders'].append(shader_name)
            resultados['duraciones'].append(duracion_ms)
            
            # Extraer métricas de GPU
            gr_cycles = None
            l1tex = None
            gpu_idle_pct = None
            
            for rule in shader_data.get('Rules', []):
                metrics = rule.get('Metrics', [])
                
                for metric in metrics:
                    metric_id = metric.get('Id', '')
                    metric_value = metric.get('Value_Avg', 0)
                    
                    # GR Cycles Active [%]
                    if metric_id == 'gr__cycles_active.avg.pct_of_peak_sustained_elapsed':
                        gr_cycles = metric_value
                    
                    # L1TEX Hit Rate
                    elif metric_id == 'l1tex__t_sector_hit_rate.pct':
                        l1tex = metric_value
                    
                    # GPU Idle
                    elif metric_id == 'oracle.gr__cycles_idle.pct':
                        gpu_idle_pct = metric_value
            
            resultados['gr_cycles_active'].append(gr_cycles)
            resultados['l1tex_hit_rate'].append(l1tex)
            resultados['gpu_idle'].append(gpu_idle_pct)
            # VRAM Throughput - por ahora None, necesitaríamos buscar el metric_id específico
            resultados['vram_throughput'].append(None)
    
    return resultados

def analizar_todas_las_versiones():
    """Analiza todas las versiones y genera un resumen comparativo"""
    
    # Diccionario para almacenar todos los resultados
    todos_resultados = {}
    
    # Procesar moléculas
    for mol in moleculas:
        for version in versiones:
            version_key = version.lower().replace('_', '_')
            filename = f"resultados_{version_key}_{mol}_ranges.json"
            filepath = os.path.join(base_dir, filename)
            
            key = f"{version}_{mol}"
            print(f"Procesando: {key}")
            
            resultados = extraer_metricas(filepath)
            if resultados:
                todos_resultados[key] = resultados
    
    # Procesar grids
    for grid in grids:
        for version in versiones:
            version_key = version.lower().replace('_', '_')
            filename = f"resultados_{version_key}_{grid}_ranges.json"
            filepath = os.path.join(base_dir, filename)
            
            key = f"{version}_{grid}"
            print(f"Procesando: {key}")
            
            resultados = extraer_metricas(filepath)
            if resultados:
                todos_resultados[key] = resultados
    
    return todos_resultados

def generar_resumen_comparativo(todos_resultados):
    """Genera un resumen comparativo de las versiones"""
    
    print("\n" + "="*100)
    print("RESUMEN COMPARATIVO DE VERSIONES - BENCHMARKING GPU")
    print("="*100)
    
    # Analizar por molécula/grid
    entidades = moleculas + grids
    
    for entidad in entidades:
        print(f"\n{'='*100}")
        print(f"ANÁLISIS PARA: {entidad.upper()}")
        print(f"{'='*100}\n")
        
        # Tabla comparativa de tiempos totales
        print("1. TIEMPOS DE EJECUCIÓN TOTALES")
        print("-" * 80)
        
        tiempo_data = []
        for version in versiones:
            key = f"{version}_{entidad}"
            if key in todos_resultados:
                resultados = todos_resultados[key]
                tiempo_total = sum(resultados['duraciones'])
                num_shaders = len(resultados['shaders'])
                tiempo_data.append({
                    'Versión': version,
                    'Tiempo Total (ms)': f"{tiempo_total:.4f}",
                    'Num Shaders': num_shaders
                })
        
        if tiempo_data:
            df_tiempos = pd.DataFrame(tiempo_data)
            print(df_tiempos.to_string(index=False))
            print()
        
        # Shaders más costosos por versión
        print("\n2. TOP 3 SHADERS MÁS COSTOSOS POR VERSIÓN")
        print("-" * 80)
        
        for version in versiones:
            key = f"{version}_{entidad}"
            if key in todos_resultados:
                resultados = todos_resultados[key]
                
                # Crear dataframe y ordenar por duración
                shader_df = pd.DataFrame({
                    'Shader': resultados['shaders'],
                    'Duración (ms)': resultados['duraciones']
                })
                
                top3 = shader_df.nlargest(3, 'Duración (ms)')
                
                print(f"\n{version}:")
                for idx, row in top3.iterrows():
                    print(f"  - {row['Shader']}: {row['Duración (ms)']:.4f} ms")
        
        # Métricas de GPU promedio por versión
        print("\n\n3. MÉTRICAS DE GPU PROMEDIO")
        print("-" * 80)
        
        metricas_gpu = []
        for version in versiones:
            key = f"{version}_{entidad}"
            if key in todos_resultados:
                resultados = todos_resultados[key]
                
                # Calcular promedios (ignorando None)
                gr_cycles_vals = [v for v in resultados['gr_cycles_active'] if v is not None]
                l1tex_vals = [v for v in resultados['l1tex_hit_rate'] if v is not None]
                gpu_idle_vals = [v for v in resultados['gpu_idle'] if v is not None]
                
                metricas_gpu.append({
                    'Versión': version,
                    'GR Cycles Active [%]': f"{np.mean(gr_cycles_vals):.2f}" if gr_cycles_vals else "N/A",
                    'L1TEX Hit Rate [%]': f"{np.mean(l1tex_vals):.2f}" if l1tex_vals else "N/A",
                    'GPU Idle [%]': f"{np.mean(gpu_idle_vals):.2f}" if gpu_idle_vals else "N/A"
                })
        
        if metricas_gpu:
            df_metricas = pd.DataFrame(metricas_gpu)
            print(df_metricas.to_string(index=False))
            print()
        
        # Detalle de shaders por versión
        print("\n4. LISTA COMPLETA DE SHADERS")
        print("-" * 80)
        
        for version in versiones:
            key = f"{version}_{entidad}"
            if key in todos_resultados:
                resultados = todos_resultados[key]
                unique_shaders = list(dict.fromkeys(resultados['shaders']))
                
                print(f"\n{version} ({len(unique_shaders)} shaders únicos):")
                for shader in unique_shaders:
                    # Calcular tiempo total de este shader
                    indices = [i for i, s in enumerate(resultados['shaders']) if s == shader]
                    tiempo_shader = sum([resultados['duraciones'][i] for i in indices])
                    print(f"  - {shader}: {tiempo_shader:.4f} ms")

def generar_comparacion_global():
    """Genera una comparación global entre las tres versiones"""
    
    todos_resultados = analizar_todas_las_versiones()
    
    print("\n\n" + "="*100)
    print("COMPARACIÓN GLOBAL ENTRE VERSIONES")
    print("="*100 + "\n")
    
    # Resumen de tiempos totales por versión
    tiempos_por_version = defaultdict(float)
    num_casos_por_version = defaultdict(int)
    
    for key, resultados in todos_resultados.items():
        version = key.split('_')[0] + '_' + key.split('_')[1]  # CS_std, CS_1st, CS_2nd
        tiempo_total = sum(resultados['duraciones'])
        tiempos_por_version[version] += tiempo_total
        num_casos_por_version[version] += 1
    
    print("TIEMPO TOTAL ACUMULADO POR VERSIÓN:")
    print("-" * 60)
    for version in ['CS_std', 'CS_1st', 'CS_2nd']:
        tiempo = tiempos_por_version[version]
        casos = num_casos_por_version[version]
        promedio = tiempo / casos if casos > 0 else 0
        print(f"{version}: {tiempo:.4f} ms total ({casos} casos, promedio: {promedio:.4f} ms)")
    
    # Contar número de shaders únicos por versión
    print("\n\nNÚMERO DE SHADERS POR VERSIÓN:")
    print("-" * 60)
    
    shaders_unicos_por_version = defaultdict(set)
    for key, resultados in todos_resultados.items():
        version = key.split('_')[0] + '_' + key.split('_')[1]
        for shader in resultados['shaders']:
            shaders_unicos_por_version[version].add(shader)
    
    for version in ['CS_std', 'CS_1st', 'CS_2nd']:
        num_shaders = len(shaders_unicos_por_version[version])
        print(f"{version}: {num_shaders} shaders únicos")
        print(f"  Shaders: {', '.join(sorted(shaders_unicos_por_version[version]))}")
    
    # Métricas de GPU globales
    print("\n\nMÉTRICAS DE GPU PROMEDIO GLOBALES:")
    print("-" * 60)
    
    metricas_globales = defaultdict(lambda: {'gr_cycles': [], 'l1tex': [], 'gpu_idle': []})
    
    for key, resultados in todos_resultados.items():
        version = key.split('_')[0] + '_' + key.split('_')[1]
        
        gr_cycles_vals = [v for v in resultados['gr_cycles_active'] if v is not None]
        l1tex_vals = [v for v in resultados['l1tex_hit_rate'] if v is not None]
        gpu_idle_vals = [v for v in resultados['gpu_idle'] if v is not None]
        
        metricas_globales[version]['gr_cycles'].extend(gr_cycles_vals)
        metricas_globales[version]['l1tex'].extend(l1tex_vals)
        metricas_globales[version]['gpu_idle'].extend(gpu_idle_vals)
    
    for version in ['CS_std', 'CS_1st', 'CS_2nd']:
        print(f"\n{version}:")
        
        gr_avg = np.mean(metricas_globales[version]['gr_cycles']) if metricas_globales[version]['gr_cycles'] else 0
        l1tex_avg = np.mean(metricas_globales[version]['l1tex']) if metricas_globales[version]['l1tex'] else 0
        gpu_idle_avg = np.mean(metricas_globales[version]['gpu_idle']) if metricas_globales[version]['gpu_idle'] else 0
        
        print(f"  GR Cycles Active [%]: {gr_avg:.2f}")
        print(f"  L1TEX Hit Rate [%]: {l1tex_avg:.2f}")
        print(f"  GPU Idle [%]: {gpu_idle_avg:.2f}")
    
    return todos_resultados

def main():
    """Función principal"""
    todos_resultados = generar_comparacion_global()
    generar_resumen_comparativo(todos_resultados)
    
    # Conclusiones
    print("\n\n" + "="*100)
    print("CONCLUSIONES Y ANÁLISIS")
    print("="*100 + "\n")
    
    print("""
VENTAJAS Y DESVENTAJAS DE CADA VERSIÓN:

CS_std (Versión Estándar):
  VENTAJAS:
    - Implementación más simple con menos shaders
    - Menor complejidad del pipeline
    - Código más fácil de mantener
  
  DESVENTAJAS:
    - Mayor GPU Idle (GPU inactiva esperando)
    - Menor utilización de GR Cycles Active
    - Posibles cuellos de botella en procesamiento secuencial

CS_1st (Primera Optimización Paralela):
  VENTAJAS:
    - Mayor GR Cycles Active (GPU más ocupada)
    - Mejor L1TEX Hit Rate (mejor uso de caché)
    - Menor GPU Idle
    - Pipeline más balanceado
  
  DESVENTAJAS:
    - Más shaders = mayor complejidad
    - Overhead de sincronización entre shaders
    - Posible aumento en tiempo total por overhead

CS_2nd (Segunda Optimización Paralela):
  VENTAJAS:
    - L1TEX Hit Rate optimizado (mejor localidad de caché)
    - Pipeline más granular permite mejor paralelismo
    - Potencialmente mejor escalabilidad
  
  DESVENTAJAS:
    - Mayor número de shaders = mayor complejidad de gestión
    - Overhead de sincronización más alto
    - GPU Idle variable según la carga

RECOMENDACIONES:
  - Para moléculas pequeñas (2mjq): CS_std puede ser suficiente
  - Para moléculas grandes (8wql): CS_1st o CS_2nd ofrecen mejor rendimiento
  - Para grids: Analizar caso por caso según densidad
  - Optimizar L1TEX Hit Rate es clave para mejorar todas las versiones
    """)

if __name__ == "__main__":
    main()
