import pandas as pd
import torch
import torch.nn as nn
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

def saat_to_dilim(saat):
    hour = int(saat.split(':')[0])
    if 0 <= hour <= 5:
        return 'Gece'
    elif 6 <= hour <= 11:
        return 'Sabah'
    elif 12 <= hour <= 17:
        return 'Ogle'
    else:
        return 'Aksam'

def test_autoencoder_zamandilimli(test_csv_path, model_path='model.pth', scaler_path='scaler.save', threshold=0.01):
    df = pd.read_csv(test_csv_path)
    df.columns = df.columns.str.strip()
    df['ZamanDilimi'] = df['Saat'].apply(saat_to_dilim)

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler_dict = joblib.load(scaler_path)
    anomalies = []

    for i, row in df.iterrows():
        dilim = row['ZamanDilimi']
        sicaklik = row['Sicaklik']
        scaler = scaler_dict[dilim]

        temp_norm = scaler.transform([[sicaklik]])
        input_tensor = torch.tensor(temp_norm, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(input_tensor).numpy()
        reconstructed_temp = scaler.inverse_transform(reconstructed)
        reconstruction_error = abs(sicaklik - reconstructed_temp[0][0])

        if reconstruction_error > threshold:
            anomalies.append((i, sicaklik, reconstructed_temp[0][0], reconstruction_error))

    print("Anomaliler:")
    for idx, original, recon, err in anomalies:
        print(f"[Satır {idx}] Gerçek: {original:.2f}, Rekonstrüksiyon: {recon:.2f}, Hata: {err:.4f}")

    return anomalies

if __name__ == "__main__":
    test_autoencoder_zamandilimli("test.csv")
