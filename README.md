# DevSquad
# Herramientas Necesarias
  node, python 
# Modelo de entrenamiento y prediccion
    pip install pandas numpy scikit-learn
# Backend
## Instalacion
    yarn install
## Ejecucion
    yarn start:dev
# Frontend
## Instalacion
    npm install
## Ejecucion
    npm run dev

# Funcionamiento
## Frontend
### Envio
  manda datos especificos de un lugar, fecha y tiempo a futuro el cual predecir
### Recibe
  Informacion predicha del lugar temperatura, humedad, etc
## Backend
### Recibe
  Recibe la informacion del frontend (datos especificos de un lugar)
### Funcionamiento Backend
  usa la api de nasa power
    https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M,WS2M&community=RE&longitude=-65.2594306&latitude=-19.0477251&start=20010101&end=20010101&format=JSON
    varia en los parametros
  genera un csv y manda al modelo
### Recibe 
  un csv con la infomracion predicha del mes escogido
### Envia
   Envia al front la informacion ya filtrada del dia y hora en particular
## Modelo
  relaiza la prediccion del mes en particular
  
