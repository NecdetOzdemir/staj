# ==================== KÜTÜPHANELER ====================
import pandas as pd  # Veri işleme için pandas kütüphanesi
import numpy as np  # Sayısal işlemler için numpy kütüphanesi
from sklearn.linear_model import LinearRegression  # Doğrusal regresyon modeli
from sklearn.metrics import r2_score  # Model başarısını ölçme metrikleri
from datetime import timedelta  # Tarih ve saat işlemleri için
import requests  # HTTP istekleri için (LLM API bağlantısı)
import json  # JSON veri işleme için

# ==================== AYARLAR ====================
ESIK_DUSEN = -0.05    # Düşen trend eşik değeri: ΔT > -0.10°C (azalmama/Artış)
ESIK_YUKSELEN = 0.20   # Yükselen trend eşik değeri: ΔT > 0.20°C (hızlı artış)
TREND_PENCERE = 5      # Trend analizi için kullanılacak veri pencere boyutu
ANOMALI_KONTROL_SAYISI = 7  # Anomali kontrolü için sonraki veri sayısı
MIN_DUSUS_SAYISI = 3   # Minimum düşüş sayısı (anomali için)
GRUP_ZAMAN_ARALIGI = 15  # Anomali gruplama için maksimum zaman farkı (dakika)

# ==================== YENİ AYARLAR ====================
ESIK_ELEKTRIK_KESINTISI = 2.5   # Elektrik kesintisi sıcaklık artış eşiği
ESIK_KAPI_ACIK_KALDI = 0.80      # Kapı açık kalma sıcaklık artış eşiği
ODA_SICAKLIGI = 20.0            # Oda sıcaklığı referans değeri
ELEKTRIK_KESINTI_DUSUS_ESIGI = 3.0  # Elektrik kesintisi sonrası düşüş eşiği
KAPI_ACIK_DUSUS_ESIGI = 0.5     # Kapı açık kalma sonrası düşüş eşiği

# ==================== LLM MODEL AYARLARI ====================
LLM_PROVIDER = "ollama"  # Kullanılacak LLM sağlayıcısı
LLM_MODEL_NAME = "llama3.1:8b-instruct-q8_0"  # Kullanılacak model adı
LLM_API_URL = "http://localhost:11434"  # Ollama API URL'si
LLM_KULLAN = True  # LLM analizi aktif/pasif durumu

# ==================== DOSYA YOLU ====================
CSV_PATH = "/home/necdet/Masaüstü/dolap/test.csv"  # CSV dosya yolu

# ==================== 1. VERİ OKUMA FONKSİYONU ====================
def veri_oku():
    """CSV dosyasını okur ve temizler"""
    print("=== 1. VERİ OKUMA ===")  # Başlık yazdır
    try:
        df = pd.read_csv(CSV_PATH)  # CSV dosyasını pandas ile oku
        print("✅ CSV dosyası başarıyla okundu")  # Başarı mesajı
        
        df['Tarih'] = pd.to_datetime(df['Tarih'])  # Tarih sütununu datetime formatına çevir
        df['Value'] = pd.to_numeric(df['Value'])  # Value sütununu sayısal değere çevir
        df_sorted = df.sort_values('Tarih').reset_index(drop=True)  # Tarihe göre sırala ve indeksi sıfırla
        
        print(f"Toplam veri noktası: {len(df_sorted)}")  # Veri sayısını yazdır
        print(f"Tarih aralığı: {df_sorted['Tarih'].min()} - {df_sorted['Tarih'].max()}")  # Tarih aralığını yazdır
        print(f"Sıcaklık aralığı: {df_sorted['Value'].min():.2f}°C - {df_sorted['Value'].max():.2f}°C")  # Sıcaklık aralığını yazdır
        
        return df_sorted  # İşlenmiş veriyi döndür
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")  # Hata mesajını yazdır
        return None  # Hata durumunda None döndür

# ==================== 2. TREND ANALİZİ FONKSİYONU ====================
def trend_belirle(index, df_data, window=TREND_PENCERE):
    """Belirli bir noktanın trendini belirler"""
    if index < window or len(df_data) < window:  # Yeterli veri yoksa
        return "bilinmiyor"  # Bilinmiyor döndür
    
    prev_values = df_data['Value'].iloc[index-window:index]  # Önceki pencere kadar veriyi al
    if len(prev_values) < 3:  # Yeterli veri yoksa
        return "bilinmiyor"  # Bilinmiyor döndür
    
    try:
        x = np.array(range(len(prev_values))).reshape(-1, 1)  # X değerlerini oluştur (zaman)
        y = np.array(prev_values)  # Y değerlerini oluştur (sıcaklık)
        
        model = LinearRegression()  # Doğrusal regresyon modeli oluştur
        model.fit(x, y)  # Modeli eğit
        trend_slope = model.coef_[0]  # Trend eğimini al
        r2 = r2_score(y, model.predict(x))  # R² skorunu hesapla
        total_change = prev_values.iloc[-1] - prev_values.iloc[0]  # Toplam değişimi hesapla
        
        min_r2 = 0.6  # Minimum R² eşiği
        min_change = 0.03  # Minimum değişim eşiği
        
        if r2 < min_r2 or abs(total_change) < min_change:  # Trend belirgin değilse
            return "sabit"  # Sabit trend döndür
        elif trend_slope < -0.003 and total_change < -min_change:  # Negatif eğim ve düşüş varsa
            return "dusen"  # Düşen trend döndür
        elif trend_slope > 0.003 and total_change > min_change:  # Pozitif eğim ve artış varsa
            return "yukselen"  # Yükselen trend döndür
        else:
            return "sabit"  # Varsayılan olarak sabit trend
        
    except:
        return "bilinmiyor"  # Hata durumunda bilinmiyor döndür

# ==================== 3. DÜŞEN TRENDDE ANOMALİ KONTROLÜ ====================
def dusen_trend_anomali_kontrol(index, df_sorted):
    """
    Düşen trendde anomali kontrolü yapar:
    - Eğer (şimdi - önceki) > -0.10 ise (azalmıyor veya artıyor)
    - Sonraki 7 değeri kontrol et
    - En az 3 tanesi bir öncekinden küçükse → kapı açılmış
    """
    if index >= len(df_sorted) - ANOMALI_KONTROL_SAYISI:  # Yeterli sonraki veri yoksa
        return False  # Anomali yok
    
    onceki_deger = df_sorted['Value'].iloc[index-1]  # Önceki değeri al
    simdiki_deger = df_sorted['Value'].iloc[index]  # Şimdiki değeri al
    fark = simdiki_deger - onceki_deger  # Farkı hesapla
    
    # KRİTİK KOŞUL: ΔT > -0.10°C (azalmıyor veya artıyor)
    if fark > ESIK_DUSEN:  # Eşik değerini kontrol et
        dusus_sayisi = 0  # Düşüş sayacını sıfırla
        
        # Sonraki 7 değeri kontrol et
        for i in range(1, ANOMALI_KONTROL_SAYISI + 1):
            if index + i >= len(df_sorted):  # Sınırı aşarsa çık
                break
                
            onceki_kontrol = df_sorted['Value'].iloc[index + i - 1]  # Önceki kontrol değeri
            simdiki_kontrol = df_sorted['Value'].iloc[index + i]  # Şimdiki kontrol değeri
            kontrol_fark = simdiki_kontrol - onceki_kontrol  # Kontrol farkı
            
            # Eğer sıcaklık düştüyse (ΔT ≤ 0)
            if kontrol_fark <= 0:  # Düşüş varsa
                dusus_sayisi += 1  # Sayacı artır
        
        # En az 3 düşüş varsa kapı açılmış
        return dusus_sayisi >= MIN_DUSUS_SAYISI  # Minimum düşüş sayısını kontrol et
    
    return False  # Anomali yok

# ==================== 4. ELEKTRİK KESİNTİSİ SÜRE HESAPLAMA ====================
def elektrik_kesintisi_suresini_hesapla(baslangic_index, df_sorted):
    """
    Elektrik kesintisi olayının süresini hesaplar:
    - Başlangıç: baslangic_index
    - Bitiş: Sıcaklık 20°C'ye ulaştıktan sonra ilk 3°C düşüş
    """
    print(f"   🔍 Elektrik kesintisi süresi hesaplanıyor (başlangıç: {df_sorted['Tarih'].iloc[baslangic_index]})")
    
    # Adım 1: 20°C'ye ulaşan ilk noktayı bul
    yirmi_derece_index = None  # 20°C indeksi
    yirmi_derece_sicaklik = None  # 20°C sıcaklığı
    
    for i in range(baslangic_index, len(df_sorted)):  # Başlangıçtan sona kadar tara
        if df_sorted['Value'].iloc[i] >= ODA_SICAKLIGI:  # 20°C'ye ulaşıldıysa
            yirmi_derece_index = i  # İndeksi kaydet
            yirmi_derece_sicaklik = df_sorted['Value'].iloc[i]  # Sıcaklığı kaydet
            print(f"   ✅ 20°C'ye ulaşıldı: {df_sorted['Tarih'].iloc[i]} ({yirmi_derece_sicaklik:.2f}°C)")
            break
    
    if yirmi_derece_index is not None:  # 20°C'ye ulaşıldıysa
        # Adım 2: 20°C'ye ulaştıktan sonra 3°C düşüşü bul
        hedef_sicaklik = yirmi_derece_sicaklik - ELEKTRIK_KESINTI_DUSUS_ESIGI  # Hedef sıcaklık
        print(f"   🔍 Hedef sıcaklık: {hedef_sicaklik:.2f}°C")
        
        for i in range(yirmi_derece_index + 1, len(df_sorted)):  # 20°C sonrasını tara
            if df_sorted['Value'].iloc[i] <= hedef_sicaklik:  # Hedefe ulaşıldıysa
                bitis_zamani = df_sorted['Tarih'].iloc[i]  # Bitiş zamanını al
                sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
                print(f"   ✅ 3°C düşüş tespit edildi: {bitis_zamani} ({df_sorted['Value'].iloc[i]:.2f}°C)")
                return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür
        
        # Eğer 3°C düşüş bulunamazsa, son veri noktasını kullan
        bitis_zamani = df_sorted['Tarih'].iloc[-1]  # Son zamanı al
        sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
        print(f"   ⚠️ 3°C düşüş bulunamadı, son veri noktası kullanılıyor: {bitis_zamani}")
        return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür
    else:
        # 20°C'ye ulaşmadı, normal yöntemle ilk düşüşü bul
        print(f"   ⚠️ 20°C'ye ulaşılamadı, normal düşüş kontrolü yapılıyor")
        for i in range(baslangic_index + 1, len(df_sorted)):  # Başlangıç sonrasını tara
            if df_sorted['Value'].iloc[i] < df_sorted['Value'].iloc[i-1]:  # Düşüş varsa
                bitis_zamani = df_sorted['Tarih'].iloc[i]  # Bitiş zamanını al
                sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
                print(f"   ✅ İlk düşüş tespit edildi: {bitis_zamani}")
                return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür
        
        # Düşüş olmazsa son noktayı kullan
        bitis_zamani = df_sorted['Tarih'].iloc[-1]  # Son zamanı al
        sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
        print(f"   ⚠️ Düşüş bulunamadı, son veri noktası kullanılıyor: {bitis_zamani}")
        return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür

# ==================== 4.1. KAPI AÇIK KALMA SÜRE HESAPLAMA ====================
def kapi_acik_kalma_suresini_hesapla(baslangic_index, df_sorted):
    """
    Kapı açık kalma olayının süresini hesaplar:
    - Başlangıç: baslangic_index (kapı ilk açık kaldığı an)
    - Bitiş: İlk belirgin sıcaklık düşüşü (kapı kapandığı an)
    """
    print(f"   🔍 Kapı açık kalma süresi hesaplanıyor (başlangıç: {df_sorted['Tarih'].iloc[baslangic_index]})")
    
    # Başlangıç sıcaklığını al
    baslangic_sicaklik = df_sorted['Value'].iloc[baslangic_index]  # Başlangıç sıcaklığı
    
    # En yüksek sıcaklığı bul (tepe noktası)
    max_sicaklik = baslangic_sicaklik  # Maksimum sıcaklık
    max_sicaklik_index = baslangic_index  # Maksimum sıcaklık indeksi
    
    for i in range(baslangic_index, min(baslangic_index + 30, len(df_sorted))):  # Sonraki 30 noktayı kontrol et
        if df_sorted['Value'].iloc[i] > max_sicaklik:  # Daha yüksek sıcaklık bulunduysa
            max_sicaklik = df_sorted['Value'].iloc[i]  # Maksimumu güncelle
            max_sicaklik_index = i  # İndeksi güncelle
    
    print(f"   📈 Tepe sıcaklık: {max_sicaklik:.2f}°C ({df_sorted['Tarih'].iloc[max_sicaklik_index]})")
    
    # Tepe noktasından sonra ilk belirgin düşüşü bul
    for i in range(max_sicaklik_index + 1, len(df_sorted)):  # Tepe sonrasını tara
        # En az 0.5°C düşüş kontrolü
        if df_sorted['Value'].iloc[i] <= max_sicaklik - KAPI_ACIK_DUSUS_ESIGI:  # Belirgin düşüş varsa
            bitis_zamani = df_sorted['Tarih'].iloc[i]  # Bitiş zamanını al
            sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
            print(f"   ✅ Kapı kapanma tespit edildi: {bitis_zamani} ({df_sorted['Value'].iloc[i]:.2f}°C)")
            return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür
        
        # Alternatif: Trend değişikliği kontrolü
        if i > max_sicaklik_index + 5:  # Tepe noktasından sonra en az 5 nokta geçti
            # Son 5 noktanın trendini kontrol et
            recent_values = df_sorted['Value'].iloc[i-5:i]  # Son 5 değeri al
            if len(recent_values) >= 3:  # Yeterli veri varsa
                x = np.array(range(len(recent_values))).reshape(-1, 1)  # X değerlerini oluştur
                y = np.array(recent_values)  # Y değerlerini oluştur
                
                model = LinearRegression()  # Model oluştur
                model.fit(x, y)  # Modeli eğit
                trend_slope = model.coef_[0]  # Trend eğimini al
                
                # Eğer trend negatifse ve belirginse
                if trend_slope < -0.01:  # Negatif eğim varsa
                    bitis_zamani = df_sorted['Tarih'].iloc[i]  # Bitiş zamanını al
                    sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
                    print(f"   ✅ Trend değişikliği tespit edildi: {bitis_zamani} (eğim: {trend_slope:.4f})")
                    return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür
    
    # Belirgin düşüş bulunamazsa, son veri noktasını kullan
    bitis_zamani = df_sorted['Tarih'].iloc[-1]  # Son zamanı al
    sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60  # Süreyi hesapla
    print(f"   ⚠️ Belirgin düşüş bulunamadı, son veri noktası kullanılıyor: {bitis_zamani}")
    return sure_dakika, bitis_zamani  # Süre ve bitiş zamanını döndür

# ==================== 5. LLM ANALİZ FONKSİYONLARI ====================
def hazirla_llm_verileri(olay, df_sorted):
    """
    LLM'e gönderilecek verileri hazırlar:
    - Olay başlangıcından önceki 10 veri
    - Olay süresince tüm veriler
    - Olay bitişinden sonraki 10 veri
    """
    baslangic_index = df_sorted[df_sorted['Tarih'] == olay['tarih']].index[0]  # Başlangıç indeksini bul
    bitis_index = df_sorted[df_sorted['Tarih'] == olay['bitis_zamani']].index[0]  # Bitiş indeksini bul
    
    # Başlangıçtan önceki 10 veri
    onceki_veriler = df_sorted.iloc[max(0, baslangic_index-10):baslangic_index]  # Önceki verileri al
    
    # Olay süresince tüm veriler
    olay_verileri = df_sorted.iloc[baslangic_index:bitis_index+1]  # Olay verilerini al
    
    # Bitişten sonraki 10 veri
    sonraki_veriler = df_sorted.iloc[bitis_index+1:min(len(df_sorted), bitis_index+11)]  # Sonraki verileri al
    
    return {  # Veri paketini döndür
        'onceki_veriler': onceki_veriler,
        'olay_verileri': olay_verileri,
        'sonraki_veriler': sonraki_veriler,
        'baslangic_zamani': olay['tarih'],
        'bitis_zamani': olay['bitis_zamani'],
        'sure_dakika': olay['sure_dakika']
    }

def llm_analiz_et(veri_paketi, olay_numarasi):
    """
    Ollama kullanarak olayı analiz eder
    """
    if not LLM_KULLAN:  # LLM kullanımı kapalıysa
        return {  # Varsayılan sonuçları döndür
            'kapi_acik_puan': 0,
            'elektrik_kesinti_puan': 0,
            'diger_puan': 0,
            'yorum': 'LLM analizi devre dışı'
        }
    
    # Verileri LLM anlayacağı formata dönüştür
    onceki_veri_str = "\n".join([f"{row['Tarih']}: {row['Value']:.2f}°C" for _, row in veri_paketi['onceki_veriler'].iterrows()])  # Önceki verileri string'e çevir
    olay_veri_str = "\n".join([f"{row['Tarih']}: {row['Value']:.2f}°C" for _, row in veri_paketi['olay_verileri'].iterrows()])  # Olay verilerini string'e çevir
    sonraki_veri_str = "\n".join([f"{row['Tarih']}: {row['Value']:.2f}°C" for _, row in veri_paketi['sonraki_veriler'].iterrows()])  # Sonraki verileri string'e çevir
    
    # LLM prompt'u oluştur (JSON formatında yanıt isteyerek)
    prompt = f"""
Sen bir buzdolabı arıza analiz uzmanısın. Aşağıdaki sıcaklık verilerini analiz ederek olayın ne olduğunu belirle.
OLAY BİLGİLERİ:
- Olay: {olay_numarasi}. olay
- Başlangıç Zamanı: {veri_paketi['baslangic_zamani']}
- Bitiş Zamanı: {veri_paketi['bitis_zamani']}
- Süre: {veri_paketi['sure_dakika']:.1f} dakika
SICAKLIK VERİLERİ:
Başlangıçtan Önceki 10 Veri:
{onceki_veri_str}
Olay Süresince Tüm Veriler:
{olay_veri_str}
Bitişten Sonraki 10 Veriler:
{sonraki_veri_str}
### OLAY TANIMLARI (Kıyaslama Tablosu):
| Senaryo                  | Artış Hızı        | Stabil Değer Aralığı                               | Düşüş Olur mu? | Açıklama                                                    |
|--------------------------|-------------------|----------------------------------------------------|----------------|--------------------------------------------------------------|
| Kapı Açık Kalma          | Hızlı artar       | 15°C – 20°C civarı                                 | Evet           | Hızlı artış, sonra sabitlenme ve düşüş                      |
| Elektrik Kesintisi       | Yavaş artar       | 25°C – 35°C                                        | Evet           | Artış uzun sürede, sonra düşüş                              |
| Diğer Durumlar           | Değişken          | Belirsiz / Sıcaklık çok yüksek, çok düşük veya dengesiz olabilir | Belirsiz       | Bilinen hiçbir senaryoya tam uymayan anomaliler (örn. fan arızası, sensör bozulması, fiziksel sorun) |
---
### GÖREVİN:
1. Her bir senaryo için **0 ile 10 arası orantılı puan** ver. Üç senaryonun puanlarının toplamı **tam olarak 10 olmalı**.
2. YORUM kısmında, **veri örüntüsünü analiz ederek** neden bu puanları verdiğini açıkla.
---
### YANITI AŞAĞIDAKİ JSON FORMATINDA VER:
{{
  "kapi_acik_puan": [0-10 arası tam sayı],
  "elektrik_kesinti_puan": [0-10 arası tam sayı],
  "diger_puan": [0-10 arası tam sayı],
  "yorum": "Açıklama"
}}
"""
    
    try:
        # Ollama API'sine istek gönder
        response = requests.post(  # POST isteği gönder
            f"{LLM_API_URL}/api/generate",  # API URL'si
            json={  # JSON payload
                "model": LLM_MODEL_NAME,  # Model adı
                "prompt": prompt,  # Prompt
                "stream": False,  # Stream değil
                "format": "json"  # JSON formatında yanıt iste
            }
        )
        
        if response.status_code == 200:  # Başarılı yanıt alındıysa
            result = response.json()  # JSON yanıtını parse et
            llm_cikti = result['response']  # Yanıtı al
            
            try:
                # JSON olarak parse et
                llm_json = json.loads(llm_cikti)  # JSON string'i parse et
                
                return {  # Sonuçları döndür
                    'kapi_acik_puan': llm_json.get('kapi_acik_puan', 0),
                    'elektrik_kesinti_puan': llm_json.get('elektrik_kesinti_puan', 0),
                    'diger_puan': llm_json.get('diger_puan', 0),
                    'yorum': llm_json.get('yorum', ''),
                    'tam_cikti': llm_cikti
                }
            except json.JSONDecodeError:  # JSON parse hatası olursa
                # JSON parse edilemezse, eski yöntemi dene
                lines = llm_cikti.split('\n')  # Satırlara böl
                kapi_puan = 0  # Varsayılan puanlar
                elektrik_puan = 0
                diger_puan = 0
                yorum = ""  # Boş yorum
                yorum_basladi = False  # Yorum başlangıç bayrağı
                
                for line in lines:  # Her satırı işle
                    # Puanları ayıkla
                    if "KAPI AÇIK KALMA OLASILIĞI" in line:  # Kapı açık puanı satırı
                        try:
                            if ":" in line:  # : varsa
                                kapi_puan = int(line.split(':')[1].strip().split()[0])  # Puanı al
                            else:  # : yoksa
                                import re  # Regex modülünü import et
                                match = re.search(r'\((\d+)\)', line)  # Parantez içindeki sayıyı ara
                                if match:  # Bulunduysa
                                    kapi_puan = int(match.group(1))  # Puanı al
                        except:  # Hata olursa
                            pass  # Atla
                    elif "ELEKTRİK KESİNTİSİ OLASILIĞI" in line:  # Elektrik kesintisi puanı satırı
                        try:
                            if ":" in line:  # : varsa
                                elektrik_puan = int(line.split(':')[1].strip().split()[0])  # Puanı al
                            else:  # : yoksa
                                import re  # Regex modülünü import et
                                match = re.search(r'\((\d+)\)', line)  # Parantez içindeki sayıyı ara
                                if match:  # Bulunduysa
                                    elektrik_puan = int(match.group(1))  # Puanı al
                        except:  # Hata olursa
                            pass  # Atla
                    elif "DİĞER DURUM OLASILIĞI" in line:  # Diğer durum puanı satırı
                        try:
                            if ":" in line:  # : varsa
                                diger_puan = int(line.split(':')[1].strip().split()[0])  # Puanı al
                            else:  # : yoksa
                                import re  # Regex modülünü import et
                                match = re.search(r'\((\d+)\)', line)  # Parantez içindeki sayıyı ara
                                if match:  # Bulunduysa
                                    diger_puan = int(match.group(1))  # Puanı al
                        except:  # Hata olursa
                            pass  # Atla
                    elif "YORUM:" in line:  # Yorum satırı
                        yorum_basladi = True  # Yorum başlangıcını işaretle
                        try:
                            yorum = line.split(':', 1)[1].strip()  # Yorumu al
                        except:  # Hata olursa
                            yorum = ""  # Boş yorum
                    elif yorum_basladi and line.strip():  # Yorum devamı ve boş değilse
                        # Yorum satırı devam ediyor olabilir
                        yorum += " " + line.strip()  # Yorumu ekle
                
                # Eğer yorum bulunamazsa, LLM çıktısının son kısmını yorum olarak kullan
                if not yorum:  # Yorum boşsa
                    kelimeler = llm_cikti.split()  # Kelimelere böl
                    if len(kelimeler) > 20:  # 20'den fazla kelime varsa
                        yorum = " ".join(kelimeler[-20:])  # Son 20 kelimeyi al
                    else:  # 20'den az kelime varsa
                        yorum = llm_cikti  # Tüm çıktıyı kullan
                
                return {  # Sonuçları döndür
                    'kapi_acik_puan': kapi_puan,
                    'elektrik_kesinti_puan': elektrik_puan,
                    'diger_puan': diger_puan,
                    'yorum': yorum,
                    'tam_cikti': llm_cikti
                }
        else:  # API hatası olursa
            return {  # Hata sonuçlarını döndür
                'kapi_acik_puan': 0,
                'elektrik_kesinti_puan': 0,
                'diger_puan': 0,
                'yorum': f'LLM API hatası: {response.status_code}',
                'hata': True
            }
    
    except Exception as e:  # Genel hata olursa
        return {  # Hata sonuçlarını döndür
            'kapi_acik_puan': 0,
            'elektrik_kesinti_puan': 0,
            'diger_puan': 0,
            'yorum': f'LLM bağlantı hatası: {str(e)}',
            'hata': True
        }

# ==================== 6. KAPI AÇILIMI ANALİZİ ====================
def kapi_acilimi_analiz(df_sorted):
    """Kapak açılımı tespiti yapar"""
    print("\n=== 2. OLAY TESPİTİ ===")  # Başlık yazdır
    
    kapi_acilimlar = []  # Olay listesi
    elektrik_kesintisi_active = False  # Elektrik kesintisi aktif mi?
    elektrik_kesintisi_bitis = None  # Elektrik kesintisi bitiş zamanı
    
    for i in range(1, len(df_sorted)):  # Tüm veri noktalarını tara
        # Eğer elektrik kesintisi dönemindeysek, atla
        if elektrik_kesintisi_active and elektrik_kesintisi_bitis and df_sorted['Tarih'].iloc[i] <= elektrik_kesintisi_bitis:
            continue  # Sonraki adıma geç
            
        onceki_deger = df_sorted['Value'].iloc[i-1]  # Önceki değeri al
        simdiki_deger = df_sorted['Value'].iloc[i]  # Şimdiki değeri al
        fark = simdiki_deger - onceki_deger  # Farkı hesapla
        
        # Trendi belirle (önceki noktanın trendi)
        trend = trend_belirle(i-1, df_sorted)  # Trendi belirle
        
        # YÜKSELEN TRENDDE OLAY TESPİTİ
        if trend == "yukselen":  # Yükselen trendse
            if fark > ESIK_YUKSELEN:  # Eşiği aştıysa
                olay_tipi = ""  # Olay tipi
                aciklama = ""  # Açıklama
                
                # Sınıflandırma mantığı
                if fark >= ESIK_ELEKTRIK_KESINTISI:  # Elektrik kesintisi eşiği
                    olay_tipi = "elektrik_kesintisi"  # Olay tipini belirle
                    aciklama = "Yükselen trendde elektrik kesintisi"  # Açıklama
                    
                    # Elektrik kesintisi süresini hesapla
                    sure_dakika, bitis_zamani = elektrik_kesintisi_suresini_hesapla(i, df_sorted)  # Süreyi hesapla
                    
                    # Elektrik kesintisi modunu aktif et
                    elektrik_kesintisi_active = True  # Aktif et
                    elektrik_kesintisi_bitis = bitis_zamani  # Bitiş zamanını kaydet
                    
                    print(f"\n⚡ ELEKTRİK KESİNTİSİ TESPİT EDİLDİ:")  # Mesaj yazdır
                    print(f"   Başlangıç: {df_sorted['Tarih'].iloc[i]}")  # Başlangıç zamanı
                    print(f"   Bitiş: {bitis_zamani}")  # Bitiş zamanı
                    print(f"   Süre: {sure_dakika:.1f} dakika")  # Süre
                    print(f"   Sıcaklık Farkı: {fark:.2f}°C")  # Sıcaklık farkı
                    
                    # Olay detaylarını ekle
                    kapi_acilimlar.append({  # Olayı listeye ekle
                        'tarih': df_sorted['Tarih'].iloc[i],
                        'tip': olay_tipi,
                        'aciklama': aciklama,
                        'fark': fark,
                        'trend': 'yukselen',
                        'bitis_zamani': bitis_zamani,
                        'sure_dakika': sure_dakika
                    })
                    
                elif fark >= ESIK_KAPI_ACIK_KALDI:  # Kapı açık kalma eşiği
                    olay_tipi = "kapi_acik_kaldi"  # Olay tipini belirle
                    aciklama = "Yükselen trendde kapı açık kaldı"  # Açıklama
                    
                    # Kapı açık kalma süresini hesapla
                    sure_dakika, bitis_zamani = kapi_acik_kalma_suresini_hesapla(i, df_sorted)  # Süreyi hesapla
                    
                    print(f"\n🚪 KAPI AÇIK KALDI TESPİT EDİLDİ:")  # Mesaj yazdır
                    print(f"   Başlangıç: {df_sorted['Tarih'].iloc[i]}")  # Başlangıç zamanı
                    print(f"   Bitiş: {bitis_zamani}")  # Bitiş zamanı
                    print(f"   Süre: {sure_dakika:.1f} dakika")  # Süre
                    print(f"   Sıcaklık Farkı: {fark:.2f}°C")  # Sıcaklık farkı
                    
                    # Olay detaylarını ekle
                    kapi_acilimlar.append({  # Olayı listeye ekle
                        'tarih': df_sorted['Tarih'].iloc[i],
                        'tip': olay_tipi,
                        'aciklama': aciklama,
                        'fark': fark,
                        'trend': 'yukselen',
                        'bitis_zamani': bitis_zamani,
                        'sure_dakika': sure_dakika
                    })
                    
                else:  # Normal kapı açılımı
                    olay_tipi = "kapi_acilimi"  # Olay tipini belirle
                    aciklama = "Yükselen trendde normal kapı açılımı"  # Açıklama
                    
                    print(f"\n🚪 KAPI AÇILIMI TESPİT EDİLDİ:")  # Mesaj yazdır
                    print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")  # Tarih
                    print(f"   Sıcaklık Farkı: {fark:.2f}°C")  # Sıcaklık farkı
                    
                    # Olay detaylarını ekle
                    kapi_acilimlar.append({  # Olayı listeye ekle
                        'tarih': df_sorted['Tarih'].iloc[i],
                        'tip': olay_tipi,
                        'aciklama': aciklama,
                        'fark': fark,
                        'trend': 'yukselen'
                    })
        
        # DÜŞEN TRENDDE KAPAK AÇILIMI
        elif trend == "dusen":  # Düşen trendse
            if dusen_trend_anomali_kontrol(i, df_sorted):  # Anomali kontrolü yap
                print(f"\n🚪 KAPI AÇILIMI TESPİT EDİLDİ (Düşen Trend):")  # Mesaj yazdır
                print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")  # Tarih
                print(f"   Sıcaklık Farkı: {fark:.2f}°C")  # Sıcaklık farkı
                
                # Olay detaylarını ekle
                kapi_acilimlar.append({  # Olayı listeye ekle
                    'tarih': df_sorted['Tarih'].iloc[i],
                    'tip': 'kapi_acilimi',
                    'aciklama': 'Düşen trendde azalmama/Artış',
                    'fark': fark,
                    'trend': 'dusen'
                })
    
    print(f"\n📊 Toplam {len(kapi_acilimlar)} olay tespit edildi")  # Toplam olay sayısını yazdır
    
    return kapi_acilimlar  # Olay listesini döndür

# ==================== 7. ANOMALİ GRUPLAMA ====================
def grupla_anomaliler(anomaliler, zaman_araligi_dakika=GRUP_ZAMAN_ARALIGI, df_sorted=None):
    """
    Anomalileri zaman bazında gruplar:
    - Birbirine yakın zamanlarda olanları tek olay olarak birleştirir
    - Arada normal değerler olan ayrık anomalileri ayrı gruplar olarak gösterir
    """
    if not anomaliler:  # Anomali yoksa
        return []  # Boş liste döndür
    
    # Anomalileri ikiye ayır: elektrik kesintisi, kapı açık kaldı ve diğerleri
    elektrik_kesintisi_olaylari = []  # Elektrik kesintisi olayları
    kapi_acik_kaldi_olaylari = []  # Kapı açık kalma olayları
    diger_olaylar = []  # Diğer olaylar
    
    for olay in anomaliler:  # Her olayı işle
        if olay['tip'] == 'elektrik_kesintisi':  # Elektrik kesintisi ise
            elektrik_kesintisi_olaylari.append(olay)  # Listeye ekle
        elif olay['tip'] == 'kapi_acik_kaldi':  # Kapı açık kaldı ise
            kapi_acik_kaldi_olaylari.append(olay)  # Listeye ekle
        else:  # Diğer olay ise
            diger_olaylar.append(olay)  # Listeye ekle
    
    # Diğer olayları zaman sırasına göre sırala
    diger_olaylar_sirali = sorted(diger_olaylar, key=lambda x: x['tarih'])  # Tarihe göre sırala
    
    # Diğer olayları grupla
    gruplar = []  # Grup listesi
    if diger_olaylar_sirali:  # Diğer olaylar varsa
        mevcut_grup = [diger_olaylar_sirali[0]]  # İlk grup
        
        for i in range(1, len(diger_olaylar_sirali)):  # Geri kalan olayları tara
            onceki_zaman = diger_olaylar_sirali[i-1]['tarih']  # Önceki zaman
            simdiki_zaman = diger_olaylar_sirali[i]['tarih']  # Şimdiki zaman
            zaman_farki = (simdiki_zaman - onceki_zaman).total_seconds() / 60  # Zaman farkını hesapla
            
            if zaman_farki <= zaman_araligi_dakika:  # Zaman aralığı içindeyse
                mevcut_grup.append(diger_olaylar_sirali[i])  # Gruba ekle
            else:  # Değilse
                gruplar.append(mevcut_grup)  # Grubu listeye ekle
                mevcut_grup = [diger_olaylar_sirali[i]]  # Yeni grup başlat
        
        gruplar.append(mevcut_grup)  # Son grubu ekle
    
    # Grup özetlerini oluştur
    grup_ozetleri = []  # Grup özetleri
    
    # Diğer olay grupları için özetler
    for grup in gruplar:  # Her grup için
        # Grubu zaman sırasına göre sırala (güvenlik için)
        grup_sirali = sorted(grup, key=lambda x: x['tarih'])  # Tarihe göre sırala
        
        # Başlangıç ve bitiş zamanlarını doğru şekilde al
        baslangic_zamani = grup_sirali[0]['tarih']  # Başlangıç zamanı
        bitis_zamani = grup_sirali[-1]['tarih']  # Bitiş zamanı
        
        # Süreyi hesapla
        sure_dakika = (bitis_zamani - baslangic_zamani).total_seconds() / 60  # Süreyi hesapla
        
        anomali_sayisi = len(grup)  # Anomali sayısı
        max_fark = max(olay['fark'] for olay in grup)  # Maksimum fark
        
        # Dominant trendi belirle
        yukselen_sayisi = sum(1 for olay in grup if olay['trend'] == 'yukselen')  # Yükselen trend sayısı
        dusen_sayisi = sum(1 for olay in grup if olay['trend'] == 'dusen')  # Düşen trend sayısı
        dominant_trend = 'yukselen' if yukselen_sayisi >= dusen_sayisi else 'dusen'  # Dominant trend
        
        grup_ozetleri.append({  # Grup özetini ekle
            'baslangic': baslangic_zamani,
            'bitis': bitis_zamani,
            'sure_dakika': sure_dakika,
            'anomali_sayisi': anomali_sayisi,
            'max_fark': max_fark,
            'dominant_trend': dominant_trend,
            'detaylar': grup
        })
    
    # Elektrik kesintisi olaylarını tek tek ekle (gruplama yapma)
    for olay in elektrik_kesintisi_olaylari:  # Her olay için
        grup_ozetleri.append({  # Grup özetini ekle
            'baslangic': olay['tarih'],
            'bitis': olay['bitis_zamani'],
            'sure_dakika': olay['sure_dakika'],
            'anomali_sayisi': 1,
            'max_fark': olay['fark'],
            'dominant_trend': 'yukselen',
            'detaylar': [olay]
        })
    
    # Kapı açık kalma olaylarını tek tek ekle (gruplama yapma)
    for olay in kapi_acik_kaldi_olaylari:  # Her olay için
        grup_ozetleri.append({  # Grup özetini ekle
            'baslangic': olay['tarih'],
            'bitis': olay['bitis_zamani'],
            'sure_dakika': olay['sure_dakika'],
            'anomali_sayisi': 1,
            'max_fark': olay['fark'],
            'dominant_trend': 'yukselen',
            'detaylar': [olay]
        })
    
    # Zaman sırasına göre sırala
    grup_ozetleri = sorted(grup_ozetleri, key=lambda x: x['baslangic'])  # Başlangıç zamanına göre sırala
    
    return grup_ozetleri  # Grup özetlerini döndür

# ==================== 8. ANA FONKSİYON ====================
def main():
    """Ana çalışma fonksiyonu"""
    print("🏠 BUZDOLABI OLAY TESPİT SİSTEMİ")  # Başlık yazdır
    print("=" * 50)  # Ayırıcı
    print(f"🤖 LLM Model: {LLM_MODEL_NAME} (Ollama)")  # Model bilgisi
    print(f"📊 LLM Analizi: {'Aktif' if LLM_KULLAN else 'Pasif'}")  # LLM durumu
    print("=" * 50)  # Ayırıcı
    
    # 1. Veriyi oku
    df = veri_oku()  # Veriyi oku
    if df is None:  # Veri okunamadıysa
        return  # Çık
    
    # 2. Kapı açılımı analizi yap
    olay_detaylari = kapi_acilimi_analiz(df)  # Olayları analiz et
    
    # 3. Anomalileri grupla
    print("\n=== OLAY GRUPLAMA ===")  # Başlık yazdır
    gruplu_sonuclar = grupla_anomaliler(olay_detaylari, df_sorted=df)  # Anomalileri grupla
    
    print(f"\n📊 Toplam {len(gruplu_sonuclar)} olay grubu tespit edildi")  # Toplam grup sayısı
    
    # 4. LLM Analizi (sadece elektrik kesintisi ve kapı açık kalma olayları için)
    if LLM_KULLAN:  # LLM kullanımı aktifse
        print("\n=== LLM ANALİZİ ===")  # Başlık yazdır
        for i, olay in enumerate(gruplu_sonuclar, 1):  # Her olay için
            olay_tipi = olay['detaylar'][0]['tip']  # Olay tipini al
            
            # Sadece elektrik kesintisi ve kapı açık kalma olayları için LLM analizi yap
            if olay_tipi in ['elektrik_kesintisi', 'kapi_acik_kaldi']:  # Belirli olay tipleri için
                print(f"\n🔍 {i}. olay LLM ile analiz ediliyor...")  # Mesaj yazdır
                
                # LLM'e gönderilecek verileri hazırla
                veri_paketi = hazirla_llm_verileri(olay['detaylar'][0], df)  # Veri paketini hazırla
                
                # LLM analizi yap
                llm_sonuclari = llm_analiz_et(veri_paketi, i)  # LLM analizi yap
                
                # Olaya LLM sonuçlarını ekle
                olay['llm_analizi'] = llm_sonuclari  # Sonuçları olaya ekle
                
                print(f"   ✅ LLM analizi tamamlandı")  # Mesaj yazdır
                print(f"   📈 Kapı Açık Puan: {llm_sonuclari['kapi_acik_puan']}/10")  # Puanı yazdır
                print(f"   ⚡ Elektrik Kesinti Puan: {llm_sonuclari['elektrik_kesinti_puan']}/10")  # Puanı yazdır
                print(f"   ❓ Diğer Puan: {llm_sonuclari['diger_puan']}/10")  # Puanı yazdır
                print(f"   💬 Yorum: {llm_sonuclari['yorum']}")  # Yorumu yazdır
    
    # 5. Gruplanmış sonuçları göster
    if gruplu_sonuclar:  # Sonuçlar varsa
        print("\n📋 TESPİT EDİLEN OLAYLAR:")  # Başlık yazdır
        print("=" * 50)  # Ayırıcı
        
        for i, olay in enumerate(gruplu_sonuclar, 1):  # Her olay için
            olay_tipi = olay['detaylar'][0]['tip']  # Olay tipini al
            
            # Olay tipine göre ikon ve başlık
            if olay_tipi == 'elektrik_kesintisi':  # Elektrik kesintisi ise
                ikon = "⚡"  # İkon
                baslik = "ELEKTRİK KESİNTİSİ"  # Başlık
            elif olay_tipi == 'kapi_acik_kaldi':  # Kapı açık kaldı ise
                ikon = "🚪"  # İkon
                baslik = "KAPI AÇIK KALDI"  # Başlık
            else:  # Diğer olaylar
                ikon = "🚪"  # İkon
                baslik = "KAPI AÇILIMI"  # Başlık
            
            print(f"\n{ikon} OLAY {i}: {baslik}")  # Olay başlığı
            print(f"   Başlangıç: {olay['baslangic'].strftime('%Y-%m-%d %H:%M:%S')}")  # Başlangıç zamanı
            print(f"   Bitiş: {olay['bitis'].strftime('%Y-%m-%d %H:%M:%S')}")  # Bitiş zamanı
            print(f"   Süre: {olay['sure_dakika']:.1f} dakika")  # Süre
            print(f"   Olay Sayısı: {olay['anomali_sayisi']}")  # Olay sayısı
            print(f"   Maksimum Sıcaklık Farkı: {olay['max_fark']:.2f}°C")  # Maksimum fark
            print(f"   Dominant Trend: {olay['dominant_trend']}")  # Dominant trend
            
            # LLM analizi varsa göster
            if 'llm_analizi' in olay:  # LLM analizi varsa
                llm = olay['llm_analizi']  # LLM sonuçlarını al
                print(f"\n   🤖 LLM ANALİZİ:")  # Başlık yazdır
                print(f"      📈 Kapı Açık Olasılığı: {llm['kapi_acik_puan']}/10")  # Puanı yazdır
                print(f"      ⚡ Elektrik Kesintisi Olasılığı: {llm['elektrik_kesinti_puan']}/10")  # Puanı yazdır
                print(f"      ❓ Diğer Durum Olasılığı: {llm['diger_puan']}/10")  # Puanı yazdır
                print(f"      💬 Yorum: {llm['yorum']}")  # Yorumu yazdır

# ==================== PROGRAMI ÇALIŞTIR ====================
if __name__ == "__main__":  # Program doğrudan çalıştırılıyorsa
    main()  # Ana fonksiyonu çağır