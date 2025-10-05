import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Obtener la fecha objetivo desde los argumentos de línea de comandos
if len(sys.argv) > 1:
    TARGET_DATE = sys.argv[1]
    print(f"🎯 Fecha objetivo recibida: {TARGET_DATE}")
else:
    print("❌ Error: No se proporcionó fecha objetivo")
    sys.exit(1)

# Configurar rutas
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_datasets = os.path.join(carpeta_actual, 'datasets')
archivo_entrada = os.path.join(carpeta_datasets, f'{TARGET_DATE}.csv')
archivo_salida = os.path.join(carpeta_datasets, f'predict_{TARGET_DATE}.csv')

# Verificar que existe la carpeta datasets
if not os.path.exists(carpeta_datasets):
    print(f"📁 Creando carpeta: {carpeta_datasets}")
    os.makedirs(carpeta_datasets)

print(f"📂 Carpeta actual: {carpeta_actual}")
print(f"📂 Carpeta datasets: {carpeta_datasets}")
print(f"📄 Archivo de entrada: {archivo_entrada}")
print(f"📄 Archivo de salida: {archivo_salida}")

# Verificar que el archivo existe
if not os.path.exists(archivo_entrada):
    print(f"❌ ERROR: No se encuentra el archivo {archivo_entrada}")
    print("💡 Asegúrate de que el archivo CSV esté en la carpeta 'python/datasets/'")
    sys.exit(1)

# Cargar datos
try:
    df = pd.read_csv(archivo_entrada)
    print(f"✅ Datos cargados exitosamente: {len(df)} registros")
except Exception as e:
    print(f"❌ Error al cargar el archivo: {e}")
    sys.exit(1)

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

print("✅ Modelos entrenados! Generando predicciones para el año solicitado...")

# Generar predicciones SOLO para el año objetivo
predicciones_año = []

# TARGET_DATE es YYYYMMDDHH - extraer año, mes y día
target_year = int(TARGET_DATE[0:4])
target_month = int(TARGET_DATE[4:6])
target_day = int(TARGET_DATE[6:8])
target_hour = int(TARGET_DATE[8:10])

import calendar

# Solo generar para el año específico que se solicita
days_in_month = calendar.monthrange(target_year, target_month)[1]

print(f"🗓️ Generando predicciones del mes {target_month:02d}/{target_year}...")

for dia in range(1, days_in_month + 1):
    fecha_dia = pd.Timestamp(year=target_year, month=target_month, day=dia)
    dia_semana = fecha_dia.dayofweek
    fin_semana = 1 if dia_semana in [5, 6] else 0

    # Mostrar progreso cada 5 días
    if dia % 5 == 0:
        print(f"📅 Generando día {dia}/{target_month:02d}/{target_year}...")

    for hora in range(24):
        fila_pred = {'fecha': f"{target_year}{target_month:02d}{dia:02d}{hora:02d}"}

        # Predecir cada variable para esta hora y día
        features_target = np.array([[dia, dia_semana, fin_semana, target_year]])

        for variable in variables:
            # Si el modelo para esta hora/variable no fue entrenado (por falta de datos), usar valor por defecto
            modelo_hora = modelos.get(variable, {}).get(hora)
            if modelo_hora is None:
                # Crear valores por defecto según variable
                if variable == 'T2M':
                    prediccion = 20.0  # temperatura promedio
                elif variable == 'PRECTOTCORR':
                    prediccion = 0.0  # sin precipitación
                elif variable == 'PS':
                    prediccion = 1013.0  # presión estándar
                elif variable == 'RH2M':
                    prediccion = 50.0  # humedad media
                elif variable == 'WS2M':
                    prediccion = 3.0  # viento moderado
                elif variable == 'ALLSKY_SFC_SW_DWN':
                    prediccion = 500.0  # radiación media
                else:  # CLOUD_AMT
                    prediccion = 50.0  # nubosidad media
            else:
                try:
                    prediccion = modelo_hora.predict(features_target)[0]
                except Exception:
                    # Valores por defecto en caso de error
                    if variable == 'T2M':
                        prediccion = 20.0
                    elif variable == 'PRECTOTCORR':
                        prediccion = 0.0
                    elif variable == 'PS':
                        prediccion = 1013.0
                    elif variable == 'RH2M':
                        prediccion = 50.0
                    elif variable == 'WS2M':
                        prediccion = 3.0
                    elif variable == 'ALLSKY_SFC_SW_DWN':
                        prediccion = 500.0
                    else:  # CLOUD_AMT
                        prediccion = 50.0

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

        predicciones_año.append(fila_pred)

# Crear DataFrame con las predicciones del año
df_predicciones = pd.DataFrame(predicciones_año)

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
    archivo_respaldo = f'predict_{TARGET_DATE}.csv'
    df_predicciones.to_csv(archivo_respaldo, index=False)
    print(f"📁 Guardado como respaldo en: {archivo_respaldo}")

print(f"\n🎉 ¡PREDICCIÓN COMPLETADA!")
print("=" * 50)
print(f"📊 Total de registros: {len(df_predicciones)}")
print(f"📅 Período: {target_year}-{target_month:02d}-01 00:00 a {target_year}-{target_month:02d}-{days_in_month} 23:00")
print(f"⏰ Horas por día: 24 | Días: {days_in_month} | Total: {len(df_predicciones)} registros")

# Buscar y mostrar la predicción específica para la fecha solicitada
prediccion_solicitada = df_predicciones[df_predicciones['fecha'] == TARGET_DATE]
if not prediccion_solicitada.empty:
    print(f"\n🎯 PREDICCIÓN PARA {TARGET_DATE}:")
    fila = prediccion_solicitada.iloc[0]
    for col in columnas_original:
        print(f"   {col}: {fila[col]}")
else:
    print(f"\n⚠️ No se encontró predicción para {TARGET_DATE}")

# Mostrar estadísticas básicas
print("\n📈 ESTADÍSTICAS DE PREDICCIÓN:")
print(f"🌡️  Temperatura (T2M): {df_predicciones['T2M'].min():.1f}°C a {df_predicciones['T2M'].max():.1f}°C")
print(f"💧 Precipitación (PRECTOTCORR): Máx {df_predicciones['PRECTOTCORR'].max():.2f} mm")
print(f"💨 Viento (WS2M): Promedio {df_predicciones['WS2M'].mean():.1f} m/s")
print(f"☁️  Nubosidad (CLOUD_AMT): Promedio {df_predicciones['CLOUD_AMT'].mean():.1f}%")