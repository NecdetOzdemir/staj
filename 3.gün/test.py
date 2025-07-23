import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# 2 saatlik zaman dilimlerine ayır
def saat_to_dilim(saat):
    hour = int(saat.split(':')[0])
    start = (hour // 2) * 2
    end = start + 1
    return f"{start:02d}-{end:02d}"

def test_autoencoder_zamandilimli(test_csv_path, model_path='model.pth', scaler_path='scaler.save', threshold_ratio=0.2):
    df = pd.read_csv(test_csv_path)
    df.columns = df.columns.str.strip()
    df['ZamanDilimi'] = df['Saat'].apply(saat_to_dilim)

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler_dict = joblib.load(scaler_path)
    anomalies = []

    real_values = []
    predicted_values = []
    timestamps = []
    anomaly_indices = []

    for i, row in df.iterrows():
        dilim = row['ZamanDilimi']
        sicaklik = row['Sicaklik']
        saat = row['Saat']
        gun = str(row['Gun']) if 'Gun' in df.columns else str(i // 24 + 1)
        label = f"{gun} {saat}"

        if dilim not in scaler_dict:
            print(f"Uyarı: {dilim} için scaler bulunamadı, atlandı.")
            continue

        scaler = scaler_dict[dilim]
        temp_norm = scaler.transform([[sicaklik]])
        input_tensor = torch.tensor(temp_norm, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(input_tensor).numpy()
        reconstructed_temp = scaler.inverse_transform(reconstructed)
        recon_val = reconstructed_temp[0][0]
        error_ratio = abs(sicaklik - recon_val) / sicaklik

        real_values.append(sicaklik)
        predicted_values.append(recon_val)
        timestamps.append(label)

        if error_ratio > threshold_ratio:
            anomaly_indices.append(i)

    # Grafik çizimi
    plt.figure(figsize=(16, 6))
    plt.plot(timestamps, real_values, color='blue', label='Gerçek Değer')
    plt.plot(timestamps, predicted_values, color='green', linestyle='dashed', label='Tahmini Değer')

    for idx in anomaly_indices:
        plt.scatter(timestamps[idx], real_values[idx], color='red', label='Anomali' if idx == anomaly_indices[0] else "")

    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Gün Saat")
    plt.ylabel("Sıcaklık (°C)")
    plt.title("Autoencoder Anomali Tespiti (2 Saatlik Dilim)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return [(idx, real_values[idx], predicted_values[idx], abs(real_values[idx] - predicted_values[idx])) for idx in anomaly_indices]

if __name__ == "__main__":
    test_autoencoder_zamandilimli("test.csv")
