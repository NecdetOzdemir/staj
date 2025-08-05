import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import timedelta

# ==================== AYARLAR ====================
ESIK_DUSEN = -0.05    # Düşen trendde: ΔT > -0.10°C (azalmama/Artış)
ESIK_YUKSELEN = 0.20   # Yükselen trendde: ΔT > 0.20°C (hızlı artış)
TREND_PENCERE = 5      # Trend analizi için pencere boyutu
ANOMALI_KONTROL_SAYISI = 7  # Sonraki kontrol edilecek değer sayısı
MIN_DUSUS_SAYISI = 3   # Minimum düşüş sayısı
GRUP_ZAMAN_ARALIGI = 15  # Gruplama için maksimum zaman farkı (dakika)

# YENİ AYARLAR
ESIK_ELEKTRIK_KESINTISI = 2.5   # Elektrik kesintisi için sıcaklık artış eşiği
ESIK_KAPI_ACIK_KALDI = 0.80      # Kapı açık kaldı için sıcaklık artış eşiği
ODA_SICAKLIGI = 20.0            # Oda sıcaklığı eşiği
ELEKTRIK_KESINTI_DUSUS_ESIGI = 3.0  # Elektrik kesintisi sonrası düşüş eşiği

# CSV dosya yolu
CSV_PATH = "/home/necdet/Masaüstü/dolap/test.csv"

# ==================== 1. VERİ OKUMA ====================
def veri_oku():
    """CSV dosyasını okur ve temizler"""
    print("=== 1. VERİ OKUMA ===")
    try:
        df = pd.read_csv(CSV_PATH)
        print("✅ CSV dosyası başarıyla okundu")
        
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df['Value'] = pd.to_numeric(df['Value'])
        df_sorted = df.sort_values('Tarih').reset_index(drop=True)
        
        print(f"Toplam veri noktası: {len(df_sorted)}")
        print(f"Tarih aralığı: {df_sorted['Tarih'].min()} - {df_sorted['Tarih'].max()}")
        print(f"Sıcaklık aralığı: {df_sorted['Value'].min():.2f}°C - {df_sorted['Value'].max():.2f}°C")
        
        return df_sorted
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        return None

# ==================== 2. TREND ANALİZİ ====================
def trend_belirle(index, df_data, window=TREND_PENCERE):
    """Belirli bir noktanın trendini belirle"""
    if index < window or len(df_data) < window:
        return "bilinmiyor"
    
    prev_values = df_data['Value'].iloc[index-window:index]
    if len(prev_values) < 3:
        return "bilinmiyor"
    
    try:
        x = np.array(range(len(prev_values))).reshape(-1, 1)
        y = np.array(prev_values)
        
        model = LinearRegression()
        model.fit(x, y)
        trend_slope = model.coef_[0]
        r2 = r2_score(y, model.predict(x))
        total_change = prev_values.iloc[-1] - prev_values.iloc[0]
        
        min_r2 = 0.6
        min_change = 0.03
        
        if r2 < min_r2 or abs(total_change) < min_change:
            return "sabit"
        elif trend_slope < -0.003 and total_change < -min_change:
            return "dusen"
        elif trend_slope > 0.003 and total_change > min_change:
            return "yukselen"
        else:
            return "sabit"
        
    except:
        return "bilinmiyor"

# ==================== 3. DÜŞEN TRENDDE ANOMALİ KONTROLÜ ====================
def dusen_trend_anomali_kontrol(index, df_sorted):
    """
    Düşen trendde anomali kontrolü:
    - Eğer (şimdi - önceki) > -0.10 ise (azalmıyor veya artıyor)
    - Sonraki 7 değeri kontrol et
    - En az 3 tanesi bir öncekinden küçükse → kapı açılmış
    - Azı düşüş gösteriyorsa → soğutucu kapanmış
    """
    if index >= len(df_sorted) - ANOMALI_KONTROL_SAYISI:
        return False
    
    onceki_deger = df_sorted['Value'].iloc[index-1]
    simdiki_deger = df_sorted['Value'].iloc[index]
    fark = simdiki_deger - onceki_deger
    
    # KRİTİK KOŞUL: ΔT > -0.10°C (azalmıyor veya artıyor)
    if fark > ESIK_DUSEN:  # ESIK_DUSEN = -0.10
        dusus_sayisi = 0
        
        # Sonraki 7 değeri kontrol et
        for i in range(1, ANOMALI_KONTROL_SAYISI + 1):
            if index + i >= len(df_sorted):
                break
                
            onceki_kontrol = df_sorted['Value'].iloc[index + i - 1]
            simdiki_kontrol = df_sorted['Value'].iloc[index + i]
            kontrol_fark = simdiki_kontrol - onceki_kontrol
            
            # Eğer sıcaklık düştüyse (ΔT ≤ 0)
            if kontrol_fark <= 0:
                dusus_sayisi += 1
        
        # En az 3 düşüş varsa kapı açılmış
        return dusus_sayisi >= MIN_DUSUS_SAYISI
    
    return False

# ==================== 4. ELEKTRİK KESİNTİSİ SÜRE HESAPLAMA ====================
def elektrik_kesintisi_suresini_hesapla(baslangic_index, df_sorted):
    """
    Elektrik kesintisi olayının süresini hesaplar:
    - Başlangıç: baslangic_index
    - Bitiş: Sıcaklık 20°C'ye ulaştıktan sonra ilk 3°C düşüş
    """
    print(f"   🔍 Elektrik kesintisi süresi hesaplanıyor (başlangıç: {df_sorted['Tarih'].iloc[baslangic_index]})")
    
    # Adım 1: 20°C'ye ulaşan ilk noktayı bul
    yirmi_derece_index = None
    yirmi_derece_sicaklik = None
    
    for i in range(baslangic_index, len(df_sorted)):
        if df_sorted['Value'].iloc[i] >= ODA_SICAKLIGI:
            yirmi_derece_index = i
            yirmi_derece_sicaklik = df_sorted['Value'].iloc[i]
            print(f"   ✅ 20°C'ye ulaşıldı: {df_sorted['Tarih'].iloc[i]} ({yirmi_derece_sicaklik:.2f}°C)")
            break
    
    if yirmi_derece_index is not None:
        # Adım 2: 20°C'ye ulaştıktan sonra 3°C düşüşü bul
        hedef_sicaklik = yirmi_derece_sicaklik - ELEKTRIK_KESINTI_DUSUS_ESIGI
        print(f"   🔍 Hedef sıcaklık: {hedef_sicaklik:.2f}°C")
        
        for i in range(yirmi_derece_index + 1, len(df_sorted)):
            if df_sorted['Value'].iloc[i] <= hedef_sicaklik:
                bitis_zamani = df_sorted['Tarih'].iloc[i]
                sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60
                print(f"   ✅ 3°C düşüş tespit edildi: {bitis_zamani} ({df_sorted['Value'].iloc[i]:.2f}°C)")
                return sure_dakika, bitis_zamani
        
        # Eğer 3°C düşüş bulunamazsa, son veri noktasını kullan
        bitis_zamani = df_sorted['Tarih'].iloc[-1]
        sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60
        print(f"   ⚠️ 3°C düşüş bulunamadı, son veri noktası kullanılıyor: {bitis_zamani}")
        return sure_dakika, bitis_zamani
    else:
        # 20°C'ye ulaşmadı, normal yöntemle ilk düşüşü bul
        print(f"   ⚠️ 20°C'ye ulaşılamadı, normal düşüş kontrolü yapılıyor")
        for i in range(baslangic_index + 1, len(df_sorted)):
            if df_sorted['Value'].iloc[i] < df_sorted['Value'].iloc[i-1]:
                bitis_zamani = df_sorted['Tarih'].iloc[i]
                sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60
                print(f"   ✅ İlk düşüş tespit edildi: {bitis_zamani}")
                return sure_dakika, bitis_zamani
        
        # Düşüş olmazsa son noktayı kullan
        bitis_zamani = df_sorted['Tarih'].iloc[-1]
        sure_dakika = (bitis_zamani - df_sorted['Tarih'].iloc[baslangic_index]).total_seconds() / 60
        print(f"   ⚠️ Düşüş bulunamadı, son veri noktası kullanılıyor: {bitis_zamani}")
        return sure_dakika, bitis_zamani

# ==================== 5. KAPI AÇILIMI ANALİZİ ====================
def kapi_acilimi_analiz(df_sorted):
    """Kapak açılımı tespiti yapar"""
    print("\n=== 2. OLAY TESPİTİ ===")
    
    kapi_acilimlar = []
    elektrik_kesintisi_active = False  # Elektrik kesintisi aktif mi?
    elektrik_kesintisi_bitis = None   # Elektrik kesintisi bitiş zamanı
    
    for i in range(1, len(df_sorted)):
        # Eğer elektrik kesintisi dönemindeysek, atla
        if elektrik_kesintisi_active and elektrik_kesintisi_bitis and df_sorted['Tarih'].iloc[i] <= elektrik_kesintisi_bitis:
            continue
            
        onceki_deger = df_sorted['Value'].iloc[i-1]
        simdiki_deger = df_sorted['Value'].iloc[i]
        fark = simdiki_deger - onceki_deger
        
        # Trendi belirle (önceki noktanın trendi)
        trend = trend_belirle(i-1, df_sorted)
        
        # YÜKSELEN TRENDDE OLAY TESPİTİ
        if trend == "yukselen":
            if fark > ESIK_YUKSELEN:
                olay_tipi = ""
                aciklama = ""
                
                # Sınıflandırma mantığı
                if fark >= ESIK_ELEKTRIK_KESINTISI:
                    olay_tipi = "elektrik_kesintisi"
                    aciklama = "Yükselen trendde elektrik kesintisi"
                    
                    # Elektrik kesintisi süresini hesapla
                    sure_dakika, bitis_zamani = elektrik_kesintisi_suresini_hesapla(i, df_sorted)
                    
                    # Elektrik kesintisi modunu aktif et
                    elektrik_kesintisi_active = True
                    elektrik_kesintisi_bitis = bitis_zamani
                    
                    print(f"\n⚡ ELEKTRİK KESİNTİSİ TESPİT EDİLDİ:")
                    print(f"   Başlangıç: {df_sorted['Tarih'].iloc[i]}")
                    print(f"   Bitiş: {bitis_zamani}")
                    print(f"   Süre: {sure_dakika:.1f} dakika")
                    print(f"   Sıcaklık Farkı: {fark:.2f}°C")
                    
                    # Olay detaylarını ekle
                    kapi_acilimlar.append({
                        'tarih': df_sorted['Tarih'].iloc[i],
                        'tip': olay_tipi,
                        'aciklama': aciklama,
                        'fark': fark,
                        'trend': 'yukselen',
                        'bitis_zamani': bitis_zamani,
                        'sure_dakika': sure_dakika
                    })
                    
                elif fark >= ESIK_KAPI_ACIK_KALDI:
                    olay_tipi = "kapi_acik_kaldi"
                    aciklama = "Yükselen trendde kapı açık kaldı"
                    
                    print(f"\n🚪 KAPI AÇIK KALDI TESPİT EDİLDİ:")
                    print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
                    print(f"   Sıcaklık Farkı: {fark:.2f}°C")
                    
                    # Olay detaylarını ekle
                    kapi_acilimlar.append({
                        'tarih': df_sorted['Tarih'].iloc[i],
                        'tip': olay_tipi,
                        'aciklama': aciklama,
                        'fark': fark,
                        'trend': 'yukselen'
                    })
                    
                else:
                    olay_tipi = "kapi_acilimi"
                    aciklama = "Yükselen trendde normal kapı açılımı"
                    
                    print(f"\n🚪 KAPI AÇILIMI TESPİT EDİLDİ:")
                    print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
                    print(f"   Sıcaklık Farkı: {fark:.2f}°C")
                    
                    # Olay detaylarını ekle
                    kapi_acilimlar.append({
                        'tarih': df_sorted['Tarih'].iloc[i],
                        'tip': olay_tipi,
                        'aciklama': aciklama,
                        'fark': fark,
                        'trend': 'yukselen'
                    })
        
        # DÜŞEN TRENDDE KAPAK AÇILIMI
        elif trend == "dusen":
            if dusen_trend_anomali_kontrol(i, df_sorted):
                print(f"\n🚪 KAPI AÇILIMI TESPİT EDİLDİ (Düşen Trend):")
                print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
                print(f"   Sıcaklık Farkı: {fark:.2f}°C")
                
                # Olay detaylarını ekle
                kapi_acilimlar.append({
                    'tarih': df_sorted['Tarih'].iloc[i],
                    'tip': 'kapi_acilimi',
                    'aciklama': 'Düşen trendde azalmama/Artış',
                    'fark': fark,
                    'trend': 'dusen'
                })
    
    print(f"\n📊 Toplam {len(kapi_acilimlar)} olay tespit edildi")
    
    return kapi_acilimlar

# ==================== 6. ANOMALİ GRUPLAMA (DÜZELTİLDİ) ====================
def grupla_anomaliler(anomaliler, zaman_araligi_dakika=GRUP_ZAMAN_ARALIGI, df_sorted=None):
    """
    Anomalileri zaman bazında gruplar:
    - Birbirine yakın zamanlarda olanları tek olay olarak birleştirir
    - Arada normal değerler olan ayrık anomalileri ayrı gruplar olarak gösterir
    """
    if not anomaliler:
        return []
    
    # Anomalileri ikiye ayır: elektrik kesintisi ve diğerleri
    elektrik_kesintisi_olaylari = []
    diger_olaylar = []
    
    for olay in anomaliler:
        if olay['tip'] == 'elektrik_kesintisi':
            elektrik_kesintisi_olaylari.append(olay)
        else:
            diger_olaylar.append(olay)
    
    # Diğer olayları zaman sırasına göre sırala
    diger_olaylar_sirali = sorted(diger_olaylar, key=lambda x: x['tarih'])
    
    # Diğer olayları grupla
    gruplar = []
    if diger_olaylar_sirali:
        mevcut_grup = [diger_olaylar_sirali[0]]
        
        for i in range(1, len(diger_olaylar_sirali)):
            onceki_zaman = diger_olaylar_sirali[i-1]['tarih']
            simdiki_zaman = diger_olaylar_sirali[i]['tarih']
            zaman_farki = (simdiki_zaman - onceki_zaman).total_seconds() / 60
            
            if zaman_farki <= zaman_araligi_dakika:
                mevcut_grup.append(diger_olaylar_sirali[i])
            else:
                gruplar.append(mevcut_grup)
                mevcut_grup = [diger_olaylar_sirali[i]]
        
        gruplar.append(mevcut_grup)
    
    # Grup özetlerini oluştur
    grup_ozetleri = []
    
    # Diğer olay grupları için özetler
    for grup in gruplar:
        # Grubu zaman sırasına göre sırala (güvenlik için)
        grup_sirali = sorted(grup, key=lambda x: x['tarih'])
        
        # Başlangıç ve bitiş zamanlarını doğru şekilde al
        baslangic_zamani = grup_sirali[0]['tarih']
        bitis_zamani = grup_sirali[-1]['tarih']
        
        # Süreyi hesapla
        sure_dakika = (bitis_zamani - baslangic_zamani).total_seconds() / 60
        
        anomali_sayisi = len(grup)
        max_fark = max(olay['fark'] for olay in grup)
        
        # Dominant trendi belirle
        yukselen_sayisi = sum(1 for olay in grup if olay['trend'] == 'yukselen')
        dusen_sayisi = sum(1 for olay in grup if olay['trend'] == 'dusen')
        dominant_trend = 'yukselen' if yukselen_sayisi >= dusen_sayisi else 'dusen'
        
        grup_ozetleri.append({
            'baslangic': baslangic_zamani,
            'bitis': bitis_zamani,
            'sure_dakika': sure_dakika,
            'anomali_sayisi': anomali_sayisi,
            'max_fark': max_fark,
            'dominant_trend': dominant_trend,
            'detaylar': grup
        })
    
    # Elektrik kesintisi olaylarını tek tek ekle (gruplama yapma)
    for olay in elektrik_kesintisi_olaylari:
        grup_ozetleri.append({
            'baslangic': olay['tarih'],
            'bitis': olay['bitis_zamani'],
            'sure_dakika': olay['sure_dakika'],
            'anomali_sayisi': 1,
            'max_fark': olay['fark'],
            'dominant_trend': 'yukselen',
            'detaylar': [olay]
        })
    
    # Zaman sırasına göre sırala
    grup_ozetleri = sorted(grup_ozetleri, key=lambda x: x['baslangic'])
    
    return grup_ozetleri

# ==================== 7. ANA FONKSİYON ====================
def main():
    """Ana çalışma fonksiyonu"""
    print("🏠 BUZDOLABI OLAY TESPİT SİSTEMİ")
    print("=" * 50)
    
    # 1. Veriyi oku
    df = veri_oku()
    if df is None:
        return
    
    # 2. Kapı açılımı analizi yap
    olay_detaylari = kapi_acilimi_analiz(df)
    
    # 3. Anomalileri grupla
    print("\n=== OLAY GRUPLAMA ===")
    gruplu_sonuclar = grupla_anomaliler(olay_detaylari, df_sorted=df)
    
    print(f"\n📊 Toplam {len(gruplu_sonuclar)} olay grubu tespit edildi")
    
    # 4. Gruplanmış sonuçları göster
    if gruplu_sonuclar:
        print("\n📋 TESPİT EDİLEN OLAYLAR:")
        print("=" * 50)
        
        for i, olay in enumerate(gruplu_sonuclar, 1):
            olay_tipi = olay['detaylar'][0]['tip']
            
            # Olay tipine göre ikon ve başlık
            if olay_tipi == 'elektrik_kesintisi':
                ikon = "⚡"
                baslik = "ELEKTRİK KESİNTİSİ"
            elif olay_tipi == 'kapi_acik_kaldi':
                ikon = "🚪"
                baslik = "KAPI AÇIK KALDI"
            else:
                ikon = "🚪"
                baslik = "KAPI AÇILIMI"
            
            print(f"\n{ikon} OLAY {i}: {baslik}")
            print(f"   Başlangıç: {olay['baslangic'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Bitiş: {olay['bitis'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Süre: {olay['sure_dakika']:.1f} dakika")
            print(f"   Olay Sayısı: {olay['anomali_sayisi']}")
            print(f"   Maksimum Sıcaklık Farkı: {olay['max_fark']:.2f}°C")
            print(f"   Dominant Trend: {olay['dominant_trend']}")

# ==================== PROGRAMI ÇALIŞTIR ====================
if __name__ == "__main__":
    main()