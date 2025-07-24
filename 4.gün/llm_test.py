# Gerekli kütüphaneler
import pandas as pd               # CSV dosyalarını okumak ve veri analizi için
import torch                      # PyTorch, derin öğrenme modeli için
import torch.nn as nn             # Sinir ağları katmanları için
import matplotlib.pyplot as plt   # Grafik çizimleri için
import joblib                     # Model scaler nesnelerini kaydetmek/yüklemek için
import subprocess                 # Terminal komutları çalıştırmak (ollama run vs.) için

# ------------------ AUTOENCODER MODELİ ------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder kısmı (veriyi sıkıştıran katmanlar)
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),   # Giriş boyutu 1 -> 16 nöron
            nn.ReLU(),          # Aktivasyon fonksiyonu
            nn.Linear(16, 8),   # 16 -> 8
            nn.ReLU(),
            nn.Linear(8, 4),    # 8 -> 4 (en küçük boyut)
        )
        # Decoder kısmı (sıkıştırılmış temsili geri açar)
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),    # 4 -> 8
            nn.ReLU(),
            nn.Linear(8, 16),   # 8 -> 16
            nn.ReLU(),
            nn.Linear(16, 1),   # 16 -> 1 (orijinal boyut)
            nn.Sigmoid()        # Çıkışı 0-1 arası normalize eder
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
        # forward fonksiyonu, encoder ve decoder işlemlerini sırayla uygular.

# ------------------ SAATİ 2 SAATLİK DİLİME AYIRMA ------------------
def saat_to_dilim(saat):
    hour = int(saat.split(':')[0])     # "12:00" -> 12 al
    start = (hour // 2) * 2            # 2 saatlik blok başlangıcı (örn: 12 -> 12, 13 -> 12)
    end = start + 1                    # bitiş
    return f"{start:02d}-{end:02d}"    # "12-13" formatında döndür

# ------------------ EĞİTİM VERİLERİNDE İLGİLİ SAATLERİ SEÇ ------------------
def get_all_related_examples(train_df, saat):
    hour = int(saat.split(':')[0])                     # Saat kısmını al
    hours_to_check = [(hour - 1) % 24, hour, (hour + 1) % 24]
    # Saatin 1 önceki, kendisi ve 1 sonraki değerini al (örneğin 12:00 için 11:00, 12:00, 13:00)

    hours_to_check = [f"{h:02d}:00" for h in hours_to_check]
    # Listeyi "11:00", "12:00" gibi string formatına dönüştür

    filtered = train_df[train_df['Saat'].isin(hours_to_check)]
    # Eğitim verisinde bu saatlere denk gelen satırları filtrele

    return "\n".join([
        f"- {row['Gün']} {row['Saat']}: Sicaklik = {row['Sicaklik']:.2f}°C"
        for _, row in filtered.iterrows()
    ])
    # Her satırı "- Gün Saat: Sicaklik = ..." formatında bir string'e dönüştür ve birleştir

# ------------------ PROMPT OLUŞTUR ------------------
def olustur_prompt(idx, sicaklik, tahmin, fark, zaman, train_df):
    gun, saat = zaman.split()                       # Zaman string'ini "Gün" ve "Saat" olarak ayır
    context = get_all_related_examples(train_df, saat)  # İlgili eğitim verilerini al
    return (f"Eğitim verisinden benzer saatlerdeki sıcaklık örnekleri:\n{context}\n\n"
            f"Şüpheli anomali verisi:\n"
            f"Veri noktası {zaman}: Sıcaklık = {sicaklik:.2f}°C, "
            f"Tahmin = {tahmin:.2f}°C, Fark = {fark:.2f}°C.\n"
            "Bu değer anormal midir? Evet ya da hayır olarak cevap ver.")

# ------------------ DEEPSEEK SORGUSU ------------------
def deepseek_sorgula(prompt):
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1', prompt],    # Terminalde "ollama run llama3.1" komutu çalıştır
            capture_output=True,                      # Çıktıyı yakala
            text=True,                                # Metin olarak döndür
            timeout=2000                              # 2000 saniye zaman aşımı
        )
        cevap = result.stdout.strip()                 # Çıktının boşluklarını temizle
        return cevap
    except Exception as e:
        return f"Hata oluştu: {e}"                   # Hata durumunda mesaj döndür

# ------------------ TEST FONKSİYONU ------------------
def test_autoencoder_ve_deepseek(
        test_csv_path, train_csv_path, 
        model_path='/home/necdet/Masaüstü/Yeni Klasör/autoencoder/model.pth',
        scaler_path='/home/necdet/Masaüstü/Yeni Klasör/autoencoder/scaler.save',
        threshold_ratio=0.2
    ):
    # Test ve eğitim verisini CSV'den oku
    df = pd.read_csv(test_csv_path)
    df.columns = df.columns.str.strip()        # Kolon isimlerindeki boşlukları kaldır
    df['ZamanDilimi'] = df['Saat'].apply(saat_to_dilim)
    # Her satıra saat dilimini ekle

    df_train = pd.read_csv(train_csv_path)
    df_train.columns = df_train.columns.str.strip()   # Eğitim verisinin kolon adlarını temizle

    # Kaydedilen Autoencoder modelini yükle
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()    # Modeli test moduna al

    # Kaydedilen scaler'ları yükle (her saat dilimi için ayrı)
    scaler_dict = joblib.load(scaler_path)

    # Sonuçları saklamak için listeler
    real_values = []        # Gerçek sıcaklık
    predicted_values = []   # Tahmin edilen sıcaklık
    timestamps = []         # Gün + Saat bilgisi
    anomaly_indices = []    # Anormal indeksler

    # ------------------ ANOMALİ HESAPLAMA ------------------
    for i, row in df.iterrows():             # Test verisindeki her satır için
        dilim = row['ZamanDilimi']           # Hangi saat diliminde olduğunu bul
        sicaklik = row['Sicaklik']           # Gerçek sıcaklık
        saat = row['Saat']                   # Saat bilgisi
        gun = str(row['Gün']) if 'Gün' in df.columns else str(i // 24 + 1)
        # Gün kolonunun var olup olmadığını kontrol et, yoksa hesapla
        label = f"{gun} {saat}"              # "8 12:00" gibi etiket oluştur

        if dilim not in scaler_dict:         # Eğer bu dilim için scaler yoksa, geç
            print(f"Uyarı: {dilim} için scaler bulunamadı, atlandı.")
            continue

        scaler = scaler_dict[dilim]                      # İlgili scaler'ı al
        temp_norm = scaler.transform([[sicaklik]])       # Sıcaklığı normalize et
        input_tensor = torch.tensor(temp_norm, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(input_tensor).numpy()  # Autoencoder tahmini
        reconstructed_temp = scaler.inverse_transform(reconstructed)
        recon_val = reconstructed_temp[0][0]             # Tahmini sıcaklık
        error_ratio = abs(sicaklik - recon_val) / sicaklik
        # Gerçek ile tahmin arasındaki hata oranı

        # Sonuçları listeye ekle
        real_values.append(sicaklik)
        predicted_values.append(recon_val)
        timestamps.append(label)

        # Hata oranı eşikten büyükse anomali kabul et
        if error_ratio > threshold_ratio:
            anomaly_indices.append(i)

    # ------------------ GRAFİK ÇİZİMİ ------------------
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

    # ------------------ DEEPSEEK SORGULAMA ------------------
    print("\nDeepSeek Anomali Kontrolü:")
    for idx in anomaly_indices:
        sicaklik = real_values[idx]
        tahmin = predicted_values[idx]
        fark = abs(sicaklik - tahmin)
        zaman = timestamps[idx]
        prompt = olustur_prompt(idx, sicaklik, tahmin, fark, zaman, df_train)

        cevap = deepseek_sorgula(prompt)
        print(f"{zaman} - Prompt: {prompt}\nModel Cevabı: {cevap}\n")

# ------------------ ANA ÇALIŞMA BLOĞU ------------------
if __name__ == "__main__":
    test_csv_path = "/home/necdet/Masaüstü/Yeni Klasör/autoencoder/test.csv"
    train_csv_path = "/home/necdet/Masaüstü/Yeni Klasör/autoencoder/weather.csv"
    test_autoencoder_ve_deepseek(test_csv_path, train_csv_path)
