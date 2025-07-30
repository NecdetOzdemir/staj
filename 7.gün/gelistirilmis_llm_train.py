import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# ----------- AYARLAR -----------
TRAIN_CSV_PATH = "/home/necdet/Masaüstü/staj-main/7.gün/weather.csv"  # Eğitim için CSV dosyası
MODEL_SAVE_FOLDER = "/home/necdet/Masaüstü/staj-main/7.gün/egitim_dosyalari"
EPOCHS = 100
BATCH_SIZE = 16  # Şimdilik tam veri ile eğitim, batch dilersen ekleyebiliriz
LEARNING_RATE = 0.001
SAAT_DILIMI = 2  # 2 saatlik zaman dilimi

# ----------- ZAMAN DİLİMİ DÖNÜŞTÜRÜCÜ -----------
def saat_to_dilim(saat, dilim=SAAT_DILIMI):
    hour = int(saat.split(':')[0])
    start = (hour // dilim) * dilim
    end = start + dilim - 1
    return f"{start:02d}-{end:02d}"

# ----------- AUTOENCODER MODELİ -----------
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

# ----------- EĞİTİM FONKSİYONU -----------
def train_autoencoder_per_timewindow(csv_path=TRAIN_CSV_PATH, save_folder=MODEL_SAVE_FOLDER, epochs=EPOCHS, dilim=SAAT_DILIMI):
    # Veriyi oku
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Zaman dilimlerini ekle
    df['ZamanDilimi'] = df['Saat'].apply(lambda s: saat_to_dilim(s, dilim))

    # Model ve scaler dosyalarının kaydedileceği klasör var mı kontrol et
    os.makedirs(save_folder, exist_ok=True)

    for zaman_dilim in sorted(df['ZamanDilimi'].unique()):
        print(f"\n➡️  '{zaman_dilim}' zaman dilimi için eğitim başlıyor...")

        # Dilime ait veriyi seç
        dilim_veri = df[df['ZamanDilimi'] == zaman_dilim]['Sicaklik'].values.reshape(-1, 1)

        if len(dilim_veri) == 0:
            print(f"⚠️ {zaman_dilim} diliminde veri yok, atlanıyor.")
            continue

        # Normalize et
        scaler = MinMaxScaler()
        dilim_norm = scaler.fit_transform(dilim_veri)

        # Tensore çevir
        data_tensor = torch.tensor(dilim_norm, dtype=torch.float32)

        # Modeli oluştur
        model = Autoencoder(input_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Eğitim döngüsü
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(data_tensor)
            loss = criterion(output, data_tensor)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

        # Model ve scaler kaydet
        model_path = os.path.join(save_folder, f"autoencoder_{zaman_dilim}.pth")
        scaler_path = os.path.join(save_folder, f"scaler_{zaman_dilim}.save")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        print(f"✅ {zaman_dilim} için model ve scaler kaydedildi.")

# ----------- ANA ÇALIŞTIRMA -----------
if __name__ == "__main__":
    train_autoencoder_per_timewindow()
