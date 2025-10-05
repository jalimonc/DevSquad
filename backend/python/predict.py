import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Configurar rutas
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_datasets = os.path.join(carpeta_actual, 'datasets')
archivo_entrada = os.path.join(carpeta_datasets, '2025101815.csv')
archivo_salida = os.path.join(carpeta_datasets, 'predicciones_enero_2026_completo.csv')

# Verificar que existe la carpeta datasets
if not os.path.exists(carpeta_datasets):
    print(f"📁 Creando carpeta: {carpeta_datasets}")
    os.makedirs(carpeta_datasets)

print(f"📂 Carpeta actual: {carpeta_actual}")
print(f"📂 Carpeta datasets: {carpeta_datasets}")
print(f"📄 Archivo de entrada: {archivo_entrada}")

# Verificar que el archivo existe
if not os.path.exists(archivo_entrada):
    print(f"❌ ERROR: No se encuentra el archivo {archivo_entrada}")
    print("💡 Asegúrate de que el archivo CSV esté en la carpeta 'python/datasets/'")
    exit()

# Cargar datos
try:
    df = pd.read_csv(archivo_entrada)
    print(f"✅ Datos cargados exitosamente: {len(df)} registros")
except Exception as e:
    print(f"❌ Error al cargar el archivo: {e}")
    exit()

df['fecha'] = pd.to_datetime(df['fecha'], format='%Y%m%d%H')

print(f"📊 Datos cargados: {len(df)} registros")
print(f"📅 Rango temporal: {df['fecha'].min()} a {df['fecha'].max()}")

# Crear características temporales
df['hora'] = df['fecha'].dt.hour
df['dia_mes'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.dayofweek
df['año'] = df['fecha'].dt.year
df['fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)

# Variables a predecir
variables = ['T2M', 'PRECTOTCORR', 'PS', 'RH2M', 'WS2M', 'ALLSKY_SFC_SW_DWN', 'CLOUD_AMT']

print("\n🎯 Entrenando modelos para cada hora del día...")

# Diccionario para guardar modelos
modelos = {var: {} for var in variables}

# Entrenar un modelo por hora (0-23) y por variable
for hora in range(24):
    print(f"🕐 Entrenando hora {hora:02d}:00...")
    
    # Filtrar datos de esta hora específica
    datos_hora = df[df['hora'] == hora].copy()
    
    for variable in variables:
        # Features: día del mes, día de semana, fin de semana, año
        X = datos_hora[['dia_mes', 'dia_semana', 'fin_semana', 'año']]
        y = datos_hora[variable]
        
        # Entrenar modelo
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=15,
            min_samples_split=5
        )
        model.fit(X, y)
        modelos[variable][hora] = model

print("✅ Modelos entrenados! Generando predicciones para ENERO 2026 completo...")

# Generar predicciones para Enero 2026 (31 días × 24 horas = 744 registros)
predicciones_2026 = []

for dia in range(1, 32):  # Días 1 al 31 de enero
    # Calcular día de la semana para 2026-01-XX (2026-01-01 es jueves)
    fecha_base = pd.Timestamp('2026-01-01')
    fecha_dia = fecha_base + pd.Timedelta(days=dia-1)
    dia_semana = fecha_dia.dayofweek
    fin_semana = 1 if dia_semana in [5, 6] else 0
    
    if dia % 5 == 0:  # Mostrar progreso cada 5 días
        print(f"📅 Generando día {dia}/01/2026...")
    
    for hora in range(24):
        fila_pred = {'fecha': f"202601{dia:02d}{hora:02d}"}
        
        # Predecir cada variable para esta hora y día
        features_2026 = np.array([[dia, dia_semana, fin_semana, 2026]])
        
        for variable in variables:
            modelo_hora = modelos[variable][hora]
            prediccion = modelo_hora.predict(features_2026)[0]
            
            # Formatear según el tipo de variable
            if variable == 'T2M':  # Temperatura
                fila_pred[variable] = round(prediccion, 1)
            elif variable == 'PRECTOTCORR':  # Precipitación
                fila_pred[variable] = max(0, round(prediccion, 2))
            elif variable in ['PS', 'RH2M']:  # Presión, Humedad
                fila_pred[variable] = round(prediccion, 2)
            elif variable == 'WS2M':  # Viento
                fila_pred[variable] = max(0, round(prediccion, 2))
            elif variable == 'ALLSKY_SFC_SW_DWN':  # Radiación solar
                fila_pred[variable] = max(0, round(prediccion, 2))
            else:  # CLOUD_AMT
                fila_pred[variable] = max(0, min(100, round(prediccion, 2)))
        
        predicciones_2026.append(fila_pred)

# Crear DataFrame con las predicciones
df_predicciones = pd.DataFrame(predicciones_2026)

# Reordenar columnas como el original
columnas_original = ['fecha', 'T2M', 'PRECTOTCORR', 'PS', 'RH2M', 'WS2M', 'ALLSKY_SFC_SW_DWN', 'CLOUD_AMT']
df_predicciones = df_predicciones[columnas_original]

# Guardar archivo
try:
    df_predicciones.to_csv(archivo_salida, index=False)
    print(f"\n✅ Archivo guardado en: {archivo_salida}")
except Exception as e:
    print(f"❌ Error al guardar: {e}")
    # Intentar guardar en carpeta actual como respaldo
    archivo_respaldo = 'predicciones_enero_2026_completo.csv'
    df_predicciones.to_csv(archivo_respaldo, index=False)
    print(f"📁 Guardado como respaldo en: {archivo_respaldo}")

print("\n🎉 ¡PREDICCIÓN COMPLETADA!")
print("=" * 50)
print(f"📊 Total de registros: {len(df_predicciones)}")
print(f"📅 Período: 2026-01-01 00:00 a 2026-01-31 23:00")
print(f"⏰ Horas por día: 24 | Días: 31 | Total: 744 registros")

# Mostrar estadísticas básicas
print("\n📈 ESTADÍSTICAS DE PREDICCIÓN:")
print(f"🌡️  Temperatura (T2M): {df_predicciones['T2M'].min():.1f}°C a {df_predicciones['T2M'].max():.1f}°C")
print(f"💧 Precipitación (PRECTOTCORR): Máx {df_predicciones['PRECTOTCORR'].max():.2f} mm")
print(f"💨 Viento (WS2M): Promedio {df_predicciones['WS2M'].mean():.1f} m/s")
print(f"☁️  Nubosidad (CLOUD_AMT): Promedio {df_predicciones['CLOUD_AMT'].mean():.1f}%")

# Mostrar ejemplo de variación diaria
print(f"\n🔍 Ejemplo - Variación horaria del 10 de Enero 2026:")
ejemplo_dia = df_predicciones[df_predicciones['fecha'].str.startswith('20260110')]
for i in [0, 6, 12, 18, 23]:  # Mostrar algunas horas clave
    fila = ejemplo_dia.iloc[i]
    print(f"   Hora {fila['fecha'][-2:]}:00 - Temp: {fila['T2M']}°C, Radiación: {fila['ALLSKY_SFC_SW_DWN']} W/m²")