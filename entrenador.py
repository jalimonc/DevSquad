import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import io
import torch.distributions as dist

# 1. Preparar datos - MEJORADO: Split por año, agrega feature hora sinusoidal
def preparar_datos(csv_path):
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y%m%d%H')
    
    cols_num = ['T2M', 'PRECTOTCORR', 'PS', 'RH2M', 'WS2M', 'ALLSKY_SFC_SW_DWN', 'CLOUD_AMT']
    
    # MEJORADO: Agrega feature sinusoidal para hora del día (captura ciclos diarios)
    df['hora_sin'] = np.sin(2 * np.pi * df['fecha'].dt.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['fecha'].dt.hour / 24)
    cols_all = cols_num + ['hora_sin', 'hora_cos']  # Input ahora 9 dims
    
    scaler = MinMaxScaler()
    df[cols_all] = scaler.fit_transform(df[cols_all])
    
    # Split: Train <2024, Test ==2024 (ajusta si tu CSV no tiene 2024)
    train_df = df[df['fecha'].dt.year < 2024]
    test_df = df[df['fecha'].dt.year == 2024]
    
    if len(train_df) < 100:
        raise ValueError("Datos de train insuficientes.")
    if len(test_df) < 50:
        print("Advertencia: Test set pequeño; precisión podría variar.")
    
    # Secuencias para train
    window_size = 24
    X_train, y_train = [], []
    for i in range(len(train_df) - window_size):
        X_train.append(train_df[cols_all].iloc[i:i+window_size].values)
        y_train.append(train_df[cols_num].iloc[i+window_size].values)  # y solo vars originales
    
    # Secuencias para test (para validación)
    X_test, y_test = [], []
    for i in range(len(test_df) - window_size):
        X_test.append(test_df[cols_all].iloc[i:i+window_size].values)
        y_test.append(test_df[cols_num].iloc[i+window_size].values)
    
    # Promedio train original (para fallback)
    train_mean_original = train_df[cols_num].mean(axis=0).values
    
    # Últimos 24h del train para inicialización futura
    last_24h_train = train_df[cols_all].iloc[-window_size:].values.reshape(1, window_size, -1)
    
    return (np.array(X_train), np.array(y_train), 
            np.array(X_test), np.array(y_test), 
            scaler, train_df['fecha'].iloc[window_size:].values, 
            train_mean_original, last_24h_train)

# 2. Dataset (sin cambios)
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Modelo LSTM - MEJORADO: Más capacidad (hidden=100, layers=3), input=9 (con horas)
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=9, hidden_size=100, num_layers=3):  # input_size=9 ahora
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, 7)  # Output solo 7 vars
        self.fc_sigma = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        mu = self.fc_mu(h_n[-1])
        log_sigma = self.fc_sigma(h_n[-1])
        sigma = torch.exp(0.5 * log_sigma)
        return mu, sigma

# 4. Entrenamiento (sin cambios, epochs=100)
def entrenar_modelo(X, y, epochs=100, batch_size=32, lr=0.001):
    dataset = WeatherDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            mu, sigma = model(batch_x)
            dists = dist.Normal(mu, sigma)
            loss = -dists.log_prob(batch_y).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Avg Loss: {total_loss / len(loader):.4f}')
    return model

# NUEVO: Evaluar precisión en test set
def evaluar_precision(model, X_test, y_test, scaler):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            input_seq = torch.tensor(X_test[i:i+1], dtype=torch.float32)  # Secuencia individual
            mu, _ = model(input_seq)
            preds.append(mu[0].numpy())
    
    # Desnormalizar
    pred_df = pd.DataFrame(np.array(preds), columns=scaler.feature_names_in_[:7])  # Solo 7 cols
    y_test_df = pd.DataFrame(y_test, columns=scaler.feature_names_in_[:7])
    
    # Inverse transform (solo para las 7 cols)
    feature_slice = scaler.feature_names_in_[:7]  # Ignora hora_sin/cos en scaler
    pred_slice = scaler.transform(pd.DataFrame(np.zeros((len(pred_df), 2)), columns=['dummy1', 'dummy2']))  # Placeholder
    full_pred = np.hstack([pred_df.values, pred_slice])  # Fake para full scaler
    mu_real = scaler.inverse_transform(full_pred)[:, :7]
    
    full_y = np.hstack([y_test_df.values, np.zeros((len(y_test_df), 2))])
    y_real = scaler.inverse_transform(full_y)[:, :7]
    
    # MAE por variable
    mae_dict = {col: mean_absolute_error(y_real[:, i], mu_real[:, i]) for i, col in enumerate(scaler.feature_names_in_[:7])}
    mae_avg = np.mean(list(mae_dict.values()))
    print("\nPrecisión en test set (último año):")
    print(pd.DataFrame([mae_dict]).T.rename(columns={0: 'MAE'}))
    print(f"MAE promedio: {mae_avg:.2f}")
    return mae_avg

# 5. Predicción - MEJORADO: Inicial con últimos 24h train, input=9 dims
def predecir_30_dias(model, scaler, mes_futuro, año_futuro, last_24h_train, num_horas=720):
    model.eval()
    mu_all = []
    fecha_actual = datetime(año_futuro, mes_futuro, 1, 0, 0)
    
    # MEJORADO: Usa últimos 24h del train como inicial (ya incluye hora_sin/cos)
    last_seq = last_24h_train.copy()  # (1,24,9)
    
    with torch.no_grad():
        for h in range(num_horas):
            input_seq = torch.tensor(last_seq, dtype=torch.float32)
            mu, _ = model(input_seq)
            mu_all.append(mu[0, :7].numpy())  # Solo 7 outputs
            
            # Autoregresión: Desliza, agrega mu (7 dims) + hora siguiente (sin/cos)
            next_hora = (fecha_actual + timedelta(hours=h)).hour
            next_sin = np.sin(2 * np.pi * next_hora / 24)
            next_cos = np.cos(2 * np.pi * next_hora / 24)
            mu_ext = np.append(mu[0, :7].numpy(), [next_sin, next_cos])  # Extiende a 9
            
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, :] = mu_ext
    
    # Desnormalizar solo mu (7 cols)
    mu_df = pd.DataFrame(np.array(mu_all), columns=scaler.feature_names_in_[:7])
    dummy = np.zeros((len(mu_df), 2))
    full_mu = np.hstack([mu_df.values, dummy])
    mu_real = scaler.inverse_transform(full_mu)[:, :7]
    
    cols_base = scaler.feature_names_in_[:7].tolist()
    fechas = [datetime(año_futuro, mes_futuro, 1) + timedelta(hours=i) for i in range(num_horas)]
    
    preds_df = pd.DataFrame({'fecha': [f.strftime('%Y%m%d%H') for f in fechas]})
    for i, col in enumerate(cols_base):
        preds_df[col] = np.round(mu_real[:, i], 2)
    
    return preds_df[['fecha'] + cols_base]

# 6. Principal
if __name__ == "__main__":
    csv_path = '2025101815.csv'
    mes_futuro = 10
    año_futuro = 2025
    
    (X_train, y_train, X_test, y_test, scaler, _, train_mean_original, last_24h_train) = preparar_datos(csv_path)
    
    model = entrenar_modelo(X_train, y_train)
    
    # NUEVO: Evaluar precisión
    mae_avg = evaluar_precision(model, X_test, y_test, scaler)
    
    # Predicción futura
    preds_df = predecir_30_dias(model, scaler, mes_futuro, año_futuro, last_24h_train)
    
    preds_df.to_csv('predicciones_octubre_2025.csv', index=False)
    print("\nPredicciones mejoradas guardadas en 'predicciones_octubre_2025.csv'")
    print("\nEjemplo primeras 5 (debería variar más en temps diarias):")
    print(preds_df.head())
    
    if mae_avg > 5:  # Ejemplo: Si MAE alto, sugiere mejora
        print("\nTip: MAE alto? Prueba más epochs (200) o datos reales de BA para calibrar.")