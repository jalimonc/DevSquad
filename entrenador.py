import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import io
import torch.distributions as dist

# 1. Preparar datos (parsea fecha YYYYMMDDHH) - AGREGADO: retorna train_mean
def preparar_datos(csv_path):
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y%m%d%H')  # Parsea el formato
    
    cols_num = ['T2M', 'RH2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'Ps', 'CLOUD_AMT']
    scaler = MinMaxScaler()
    df[cols_num] = scaler.fit_transform(df[cols_num])
    
    # AGREGADO: Promedio histórico por columna (en escala original para inicializar)
    train_mean_original = df[cols_num].mean(axis=0).values  # Promedios del dataset completo
    
    window_size = 24
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[cols_num].iloc[i:i+window_size].values)
        y.append(df[cols_num].iloc[i+window_size].values)
    return np.array(X), np.array(y), scaler, df['fecha'].iloc[window_size:].values, train_mean_original

# 2. Dataset (sin cambios)
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Modelo LSTM (sin cambios)
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, 7)
        self.fc_sigma = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        mu = self.fc_mu(h_n[-1])
        log_sigma = self.fc_sigma(h_n[-1])
        sigma = torch.exp(0.5 * log_sigma)
        return mu, sigma

# 4. Entrenamiento (sin cambios)
def entrenar_modelo(X, y, epochs=50, batch_size=32, lr=0.001):
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

# 5. Predicción - SIMPLIFICADO: Solo usa mu (valor principal), sin rangos
def predecir_30_dias(model, scaler, mes_futuro, año_futuro, train_mean_original, num_horas=720):
    model.eval()
    mu_all = []  # Solo mu
    fecha_actual = datetime(año_futuro, mes_futuro, 1, 0, 0)
    
    # Escala el promedio histórico y repítelo para las 24h iniciales
    df_mean = pd.DataFrame([train_mean_original], columns=scaler.feature_names_in_)
    scaled_mean = scaler.transform(df_mean).flatten()  # (7,)
    last_seq = np.tile(scaled_mean, (24, 1)).reshape(1, 24, 7)  # (1,24,7) con promedios repetidos
    
    with torch.no_grad():
        for h in range(num_horas):
            input_seq = torch.tensor(last_seq, dtype=torch.float32)
            mu, _ = model(input_seq)  # Ignora sigma aquí
            mu_all.append(mu[0].numpy())
            
            # Autoregresión: Desliza y agrega la predicción (media)
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, :] = mu[0].numpy()
    
    # Desnormalizar solo mu
    mu_df = pd.DataFrame(np.array(mu_all), columns=scaler.feature_names_in_)
    mu_df = scaler.inverse_transform(mu_df)
    
    # DataFrame solo con valores principales
    cols_base = scaler.feature_names_in_.tolist()
    fechas = [datetime(año_futuro, mes_futuro, 1) + timedelta(hours=i) for i in range(num_horas)]
    
    preds_df = pd.DataFrame({'fecha': [f.strftime('%Y%m%d%H') for f in fechas]})  # Formato YYYYMMDDHH para salida
    for i, col in enumerate(cols_base):
        preds_df[col] = np.round(mu_df[:, i], 2)
    
    return preds_df[['fecha'] + cols_base]

# 6. Principal - AGREGADO: Pasa train_mean a predict
if __name__ == "__main__":
    csv_path = 'datos_filtrados_octubre.csv'
    mes_futuro = 10
    año_futuro = 2025
    
    X, y, scaler, _, train_mean_original = preparar_datos(csv_path)  # Recibe el nuevo return
    if len(X) < 100:
        raise ValueError("Datos insuficientes.")
    
    model = entrenar_modelo(X, y)
    preds_df = predecir_30_dias(model, scaler, mes_futuro, año_futuro, train_mean_original)
    
    preds_df.to_csv('predicciones_octubre_2025.csv', index=False)
    print("Predicciones guardadas en 'predicciones_octubre_2025.csv'")
    print("\nEjemplo de las primeras 5 predicciones (solo valores aproximados):")
    print(preds_df.head())