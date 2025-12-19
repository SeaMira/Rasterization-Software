import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean

# Configuración de estilo para gráficos profesionales
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class AnalizadorGPU:
    def __init__(self, base_path):
        self.base_path = base_path
        self.df_rangos = pd.DataFrame() # Datos generales de los Shaders (Tiempos)
        self.df_metricas = pd.DataFrame() # Métricas detalladas (SM, Cache, etc.)

    def cargar_tiempos_referencia(self, ruta_txt):
        """Lee el archivo de tiempos frames y devuelve lista de listas"""
        bloques = []
        if os.path.exists(ruta_txt):
            with open(ruta_txt, 'r') as f:
                content = f.read().strip()
            parts = content.split('------')
            print(parts)
            for p in parts:
                if p.strip():
                    bloques.append([float(x) for x in p.strip().splitlines() if x.strip()])
        return bloques

    def procesar_datos(self, versiones, rangos, moleculas=["1aga"]):
        """
        Recorre todas las carpetas, cruza YAML con TXT de tiempos y genera DataFrames.
        """
        records_rangos = []
        records_metricas = []
        
        # Iterar Versiones (CS_1st, CS_2nd, etc.)
        for ver in versiones:
            # Iterar Rangos (range1, range2...)
            for r_idx, rango_name in enumerate(rangos): 
                # Cargar el archivo de tiempos FRAME correspondiente a este rango y versión?
                # NOTA: Según tu descripción, parece que hay un .txt por rango. 
                # Ajusta esta ruta según donde tengas los .txt de frames
                path_frames = os.path.join(self.base_path, ver, "1aga", "YAML", f"{rango_name}_frames.txt")
                
                # Si no está en la carpeta de la versión, quizás está en la raíz por rango:
                if not os.path.exists(path_frames):
                     path_frames = os.path.join(self.base_path, f"{rango_name}_frames.txt")

                tiempos_ref = self.cargar_tiempos_referencia(path_frames)
                
                # Iterar las 3 repeticiones (archivos yaml)
                for i in range(1, 4):
                    # Ajusta la ruta a tu estructura real
                    # Ej: ../CS_1st/1aga/YAML/range1_1_analysis.yaml
                    yaml_name = f"{rango_name}_{i}_analysis.yaml"
                    yaml_path = os.path.join(self.base_path, ver, "1aga", "YAML", yaml_name)
                    
                    if not os.path.exists(yaml_path):
                        # Intento con extensión .txt si falló .yaml
                        yaml_path = yaml_path.replace('.yaml', '.txt')
                    
                    if os.path.exists(yaml_path) and i <= len(tiempos_ref):
                        with open(yaml_path, 'r', encoding='utf-8') as f:
                            data_yaml = yaml.safe_load(f)
                        
                        if not isinstance(data_yaml, list): continue

                        frame_times_bloque = tiempos_ref[i-1] # Tiempos del bloque i
                        
                        # Procesar cada Shader dentro del YAML
                        for idx_tupla, entrada in enumerate(data_yaml):
                            idx_frame = idx_tupla // 5 # 5 Shaders por frame
                            if idx_frame >= len(frame_times_bloque): break
                            
                            shader_name = entrada.get('Range', 'Unknown')
                            rel_duration = entrada.get('RelativeFrameDuration', 0)
                            
                            # CÁLCULO CRÍTICO: Tiempo Real (ms)
                            total_frame_ms = frame_times_bloque[idx_frame]
                            real_shader_ms = total_frame_ms * rel_duration

                            # Guardar info básica del Shader
                            records_rangos.append({
                                'Version': ver,
                                'Range_Dist': rango_name,
                                'Repetition': i,
                                'Shader': shader_name,
                                'Real_Time_ms': real_shader_ms,
                                'Total_Frame_ms': total_frame_ms
                            })

                            # Extraer Métricas Específicas (Solo las importantes)
                            # Buscamos métricas dentro de las reglas
                            for rule in entrada.get('Rules', []):
                                for metric in rule.get('Metrics', []):
                                    records_metricas.append({
                                        'Version': ver,
                                        'Range_Dist': rango_name,
                                        'Shader': shader_name,
                                        'Metric_Name': metric.get('Name'),
                                        'Metric_Id': metric.get('Id'),
                                        'Value': metric.get('Value_Avg', metric.get('Value', 0))
                                    })

        self.df_rangos = pd.DataFrame(records_rangos)
        self.df_metricas = pd.DataFrame(records_metricas)
        print(f"Procesamiento completo: {len(self.df_rangos)} entradas de shaders procesadas.")

    # --- ANÁLISIS 1: COMPARACIÓN TOTAL (Frame Time) ---
    def plot_comparacion_total(self):
        """Gráfico de líneas comparando el tiempo total del frame entre versiones"""
        # Agrupamos por Versión y Rango, promediando las repeticiones
        df_summary = self.df_rangos.groupby(['Version', 'Range_Dist', 'Repetition'])['Real_Time_ms'].sum().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_summary, x='Range_Dist', y='Real_Time_ms', hue='Version', marker='o', linewidth=2.5)
        plt.title('Comparación de Tiempo Total de Frame por Distancia (Menos es mejor)', fontsize=15)
        plt.ylabel('Tiempo Total (ms)')
        plt.xlabel('Distancia de Cámara')
        plt.show()

    # --- ANÁLISIS 2: TIEMPO POR SHADER (Stacked Bar) ---
    def plot_breakdown_shaders(self, version_interes):
        """Muestra en qué gasta tiempo una versión específica"""
        df_v = self.df_rangos[self.df_rangos['Version'] == version_interes]
        
        # Promedio de las repeticiones
        df_avg = df_v.groupby(['Range_Dist', 'Shader'])['Real_Time_ms'].mean().reset_index()
        
        # Pivotar para gráfico de barras apiladas
        df_pivot = df_avg.pivot(index='Range_Dist', columns='Shader', values='Real_Time_ms')
        
        df_pivot.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
        plt.title(f'Desglose de Tiempo por Shader - {version_interes}', fontsize=15)
        plt.ylabel('Tiempo (ms)')
        plt.xlabel('Distancia')
        plt.legend(title='Shaders', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # --- ANÁLISIS 3: METRICAS PROFUNDAS (Hardware) ---
    def plot_comparativa_metrica_hardware(self, shader_name, metric_name_keyword):
        """
        Compara una métrica específica de hardware entre versiones para un Shader dado.
        Ej: metric_name_keyword='SM Warp Occupancy'
        """
        # Filtramos métricas que contengan la palabra clave (ej: "L1TEX Hit Rate")
        mask_shader = self.df_metricas['Shader'] == shader_name
        mask_metric = self.df_metricas['Metric_Name'].str.contains(metric_name_keyword, case=False, na=False)
        
        df_filtered = self.df_metricas[mask_shader & mask_metric].copy()
        
        if df_filtered.empty:
            print(f"No se encontraron datos para {metric_name_keyword} en {shader_name}")
            return

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_filtered, x='Range_Dist', y='Value', hue='Version', errorbar=None)
        
        titulo_metrica = df_filtered['Metric_Name'].iloc[0] # Tomamos el nombre completo de la primera coincidencia
        plt.title(f'Comparativa: {titulo_metrica} en "{shader_name}"', fontsize=15)
        plt.ylabel('Valor Métrica')
        plt.show()

# --- EJECUCIÓN ---
# Ajusta estas variables
PATH_BASE = '../nsight_gr_results/' # Tu ruta raíz
VERSIONES = ['CS_1st', 'CS_2nd', 'CS_std']
RANGOS = ['range1', 'range2', 'range3', 'range4', 'range5']

analisis = AnalizadorGPU(PATH_BASE)

# 1. Procesar todo (Asegúrate de tener los archivos rangeX_frames.txt en su lugar)
# NOTA: Para probar, puedes comentar esto si no tienes los archivos listos y usar datos simulados
analisis.procesar_datos(VERSIONES, RANGOS)

# Si tienes datos cargados:
if not analisis.df_rangos.empty:
    
    # A. Ver quién gana (Tiempo total)
    analisis.plot_comparacion_total()
    
    # B. Ver composición interna de CS_1st
    analisis.plot_breakdown_shaders('CS_1st')
    analisis.plot_breakdown_shaders('CS_2nd')
    analisis.plot_breakdown_shaders('CS_std')
    
    # C. ¿Por qué una es más rápida? Comparar métricas de hardware
    # Ejemplo: Comparar Cache Hit Rate en el shader más pesado (asumiendo que se llama 'Cleaning Shader')
    analisis.plot_comparativa_metrica_hardware('Cleaning Shader', 'L1TEX Hit Rate')
    
    # Ejemplo: Comparar Ocupación de SM
    analisis.plot_comparativa_metrica_hardware('Cleaning Shader', 'Active Warps per Cycle')