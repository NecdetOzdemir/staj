import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# ======================= AYARLAR ========================
TRAIN_CSV_PATH = "/home/necdet/Masaüstü/Yeni Klasör/egitim_dosyalari/weather.csv"
MODEL_SAVE_FOLDER = "/home/necdet/Masaüstü/Yeni Klasör/egitim_dosyalari"
EPOCHS = 100
SAAT_DILIMI = 2  # 2 saatlik zaman dilimi

# ================== YARDIMCI FONKSİYON ====================

def saat_to_dilim(saat, dilim=SAAT_DILIMI):
    hour = int(saat.split(':')[0])
    start = (hour // dilim) * dilim
    end = start + dilim - 1
    return f"{start:02d}-{end:02d}"

# =================== AUTOENCODER SINIFI ===================

class Autoencoder(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =================== EĞİTİM FONKSİYONU ====================

def train_autoencoder_zamandilimli(train_csv_path=TRAIN_CSV_PATH, 
                                   model_save_folder=MODEL_SAVE_FOLDER, 
                                   epochs=EPOCHS,
                                   saat_dilimi=SAAT_DILIMI):

    df = pd.read_csv(train_csv_path)
    df.columns = df.columns.str.strip()
    df['ZamanDilimi'] = df['Saat'].apply(lambda s: saat_to_dilim(s, saat_dilimi))

    model = Autoencoder(input_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler_dict = {}
    losses = []

    for dilim in sorted(df['ZamanDilimi'].unique()):
        temps = df.loc[df['ZamanDilimi'] == dilim, 'Sicaklik'].values.reshape(-1, 1)
        if temps.size == 0:
            continue  # veri yoksa atla

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

    # Klasör yoksa oluştur
    os.makedirs(model_save_folder, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(train_csv_path))[0]
    model_path = os.path.join(model_save_folder, f"{model_name}_autoencoder.pth")
    scaler_path = os.path.join(model_save_folder, f"{model_name}_scalers.save")

    # Dosyaları kaydet
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_dict, scaler_path)

    print(f"\n✅ Eğitim tamamlandı.")
    print(f"Model kaydedildi: {model_path}")
    print(f"Scaler'lar kaydedildi: {scaler_path}")

    return model, scaler_dict, losses

# =================== ANA ÇALIŞTIRMA =======================

if __name__ == "__main__":
    train_autoencoder_zamandilimli()
