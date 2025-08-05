import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# 2 saatlik zaman dilimi: 00-01, 02-03, ..., 22-23
def saat_to_dilim(saat):
    hour = int(saat.split(':')[0])
    start = (hour // 2) * 2
    end = start + 1
    return f"{start:02d}-{end:02d}"

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

def train_autoencoder_zamandilimli(train_csv_path, model_save_path='model.pth', scaler_save_path='scaler.save', epochs=100):
    df = pd.read_csv(train_csv_path)
    df.columns = df.columns.str.strip()
    
    # Zaman dilimlerini 2 saatlik bloklara ayır
    df['ZamanDilimi'] = df['Saat'].apply(saat_to_dilim)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler_dict = {}
    losses = []

    for dilim in sorted(df['ZamanDilimi'].unique()):
        temps = df.loc[df['ZamanDilimi'] == dilim, 'Sicaklik'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        temps_norm = scaler.fit_transform(temps)
        scaler_dict[dilim] = scaler

        data = torch.tensor(temps_norm, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

        print(f"Dilim: {dilim} - Son Epoch Loss: {loss.item():.6f}")
        losses.append((dilim, loss.item()))

    # Scaler'ları kaydet
    joblib.dump(scaler_dict, scaler_save_path)
    # Modeli kaydet
    torch.save(model.state_dict(), model_save_path)

    print("Model ve 2 saatlik zaman dilimi scaler'ları kaydedildi.")
    return model, scaler_dict, losses

if __name__ == "__main__":
    train_autoencoder_zamandilimli('weather.csv')
