import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# ==================== AYARLAR ====================
ESIK_DUSEN = -0.10      # Düşen trendde: ΔT > -0.10°C (azalmama/Artış)
ESIK_YUKSELEN = 0.20   # Yükselen trendde: ΔT > 0.20°C (hızlı artış)
TREND_PENCERE = 5      # Trend analizi için pencere boyutu
ANOMALI_KONTROL_SAYISI = 7  # Sonraki kontrol edilecek değer sayısı
MIN_DUSUS_SAYISI = 3   # Minimum düşüş sayısı
GRUP_ZAMAN_ARALIGI = 15  # Gruplama için maksimum zaman farkı (dakika)

# CSV dosya yolu
CSV_PATH = "/home/necdet/Masaüstü/dolap/temp.csv"

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

# ==================== 4. KAPI AÇILIMI ANALİZİ ====================
def kapi_acilimi_analiz(df_sorted):
    """Kapak açılımı tespiti yapar"""
    print("\n=== 2. KAPI AÇILIMI ANALİZİ ===")
    
    kapi_acilimlar = []
    
    for i in range(1, len(df_sorted)):
        onceki_deger = df_sorted['Value'].iloc[i-1]
        simdiki_deger = df_sorted['Value'].iloc[i]
        fark = simdiki_deger - onceki_deger
        
        # Trendi belirle (önceki noktanın trendi)
        trend = trend_belirle(i-1, df_sorted)
        
        # YÜKSELEN TRENDDE KAPAK AÇILIMI
        if trend == "yukselen":
            if fark > ESIK_YUKSELEN:
                print(f"\n🔍 Yükselen trendde anomali tespit edildi:")
                print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
                print(f"   {onceki_deger:.2f}°C → {simdiki_deger:.2f}°C (fark: {fark:.2f}°C)")
                print(f"   → 🚪 KAPI AÇILDI (Soğutucu kapalıyken hızlı ısınma)")
                
                kapi_acilimlar.append({
                    'tarih': df_sorted['Tarih'].iloc[i],
                    'tip': 'kapi_acilimi',
                    'aciklama': 'Yükselen trendde hızlı artış',
                    'fark': fark,
                    'trend': 'yukselen'
                })
        
        # DÜŞEN TRENDDE KAPAK AÇILIMI
        elif trend == "dusen":
            if dusen_trend_anomali_kontrol(i, df_sorted):
                print(f"\n🔍 Düşen trendde anomali tespit edildi:")
                print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
                print(f"   {onceki_deger:.2f}°C → {simdiki_deger:.2f}°C (fark: {fark:.2f}°C)")
                print(f"   → 🚪 KAPI AÇILDI (Soğutucu açıkken azalmama/Artış)")
                
                kapi_acilimlar.append({
                    'tarih': df_sorted['Tarih'].iloc[i],
                    'tip': 'kapi_acilimi',
                    'aciklama': 'Düşen trendde azalmama/Artış',
                    'fark': fark,
                    'trend': 'dusen'
                })
            elif fark > ESIK_DUSEN:  # ΔT > -0.10 ama sonraki 7'de <3 düşüş
                print(f"\n🔍 Düşen trendde soğutucu kapanması tespit edildi:")
                print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
                print(f"   {onceki_deger:.2f}°C → {simdiki_deger:.2f}°C (fark: {fark:.2f}°C)")
                print(f"   → ❄️ SOĞUTUCU KAPANDI (Normal durum)")
    
    print(f"\n📊 TOPLAM ANOMALİ: {len(kapi_acilimlar)}")
    
    return kapi_acilimlar

# ==================== 5. ANOMALİ GRUPLAMA ====================
def grupla_anomaliler(anomaliler, zaman_araligi_dakika=GRUP_ZAMAN_ARALIGI):
    """
    Anomalileri zaman bazında gruplar:
    - Birbirine yakın zamanlarda olanları tek olay olarak birleştirir
    - Arada normal değerler olan ayrık anomalileri ayrı gruplar olarak gösterir
    """
    if not anomaliler:
        return []
    
    # Anomalileri zaman sırasına göre sırala
    sirali_anomaliler = sorted(anomaliler, key=lambda x: x['tarih'])
    
    gruplar = []
    mevcut_grup = [sirali_anomaliler[0]]
    
    for i in range(1, len(sirali_anomaliler)):
        onceki_zaman = sirali_anomaliler[i-1]['tarih']
        simdiki_zaman = sirali_anomaliler[i]['tarih']
        zaman_farki = (simdiki_zaman - onceki_zaman).total_seconds() / 60
        
        if zaman_farki <= zaman_araligi_dakika:
            mevcut_grup.append(sirali_anomaliler[i])
        else:
            gruplar.append(mevcut_grup)
            mevcut_grup = [sirali_anomaliler[i]]
    
    gruplar.append(mevcut_grup)
    
    # Grup özetlerini oluştur
    grup_ozetleri = []
    for grup in gruplar:
        baslangic_zamani = min(olay['tarih'] for olay in grup)
        bitis_zamani = max(olay['tarih'] for olay in grup)
        sure_dakika = (bitis_zamani - baslangic_zamani).total_seconds() / 60
        anomali_sayisi = len(grup)
        
        # En yüksek farkı bul
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
    
    return grup_ozetleri

# ==================== 6. GRAFİK ÇİZİMİ ====================
def grafigi_ciz(df, gruplu_sonuclar):
    """Sıcaklık verilerini ve kapı açılımı olaylarını grafikle gösterir"""
    print("\n=== 4. GRAFİK ÇİZİMİ ===")
    
    # Grafik boyutlarını ayarla
    plt.figure(figsize=(16, 8))
    
    # Ana sıcaklık grafiğini çiz
    plt.plot(df['Tarih'], df['Value'], 'b-', linewidth=1.5, label='Sıcaklık')
    
    # Kapı açılımı gruplarını renkli bölgelerle göster
    renkler = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, olay in enumerate(gruplu_sonuclar):
        renk = renkler[i % len(renkler)]
        
        # Başlangıç ve bitiş zamanlarını al
        baslangic = olay['baslangic']
        bitis = olay['bitis']
        
        # Dikey çizgiler çiz
        plt.axvline(x=baslangic, color=renk, linestyle='--', alpha=0.7, linewidth=2)
        plt.axvline(x=bitis, color=renk, linestyle='--', alpha=0.7, linewidth=2)
        
        # Arka planı renklendir
        plt.axvspan(baslangic, bitis, alpha=0.2, color=renk)
        
        # Grup etiketi ekle
        orta_zaman = baslangic + (bitis - baslangic) / 2
        max_temp = df['Value'].max()
        plt.text(orta_zaman, max_temp * 0.95, f'OLAY {i+1}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=renk, alpha=0.7))
        
        # Detaylı bilgi ekle
        info_text = f'{olay["sure_dakika"]:.1f}dk\n{olay["anomali_sayisi"]} anomali\nΔT={olay["max_fark"]:.2f}°C'
        plt.text(orta_zaman, max_temp * 0.85, info_text, 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Grafik özelliklerini ayarla
    plt.title('Buzdolabı Sıcaklığı ve Kapı Açılımı Olayları', fontsize=16, fontweight='bold')
    plt.xlabel('Zaman', fontsize=12)
    plt.ylabel('Sıcaklık (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # X eksenini formatla
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    # Legend ekle
    plt.legend(loc='upper left')
    
    # Grafik göster
    plt.tight_layout()
    plt.show()
    
    # İkinci grafik: Sadece anomali bölgelerini detaylı göster
    if gruplu_sonuclar:
        plt.figure(figsize=(16, 10))
        
        for i, olay in enumerate(gruplu_sonuclar):
            renk = renkler[i % len(renkler)]
            baslangic = olay['baslangic']
            bitis = olay['bitis']
            
            # Her olay için subplot oluştur
            plt.subplot(len(gruplu_sonuclar), 1, i+1)
            
            # Olay öncesi ve sonrası verileri al (30 dakika öncesi ve sonrası)
            baslangic_genis = baslangic - timedelta(minutes=30)
            bitis_genis = bitis + timedelta(minutes=30)
            
            mask = (df['Tarih'] >= baslangic_genis) & (df['Tarih'] <= bitis_genis)
            df_olay = df[mask]
            
            # Sıcaklık grafiğini çiz
            plt.plot(df_olay['Tarih'], df_olay['Value'], 'b-', linewidth=2, label='Sıcaklık')
            
            # Anomali bölgesini renklendir
            plt.axvspan(baslangic, bitis, alpha=0.3, color=renk)
            
            # Anomali noktalarını işaretle
            for detay in olay['detaylar']:
                plt.scatter(detay['tarih'], df[df['Tarih'] == detay['tarih']]['Value'].values[0], 
                           color=renk, s=100, zorder=5)
            
            # Başlık ve etiketler
            plt.title(f'OLAY {i+1}: {baslangic.strftime("%H:%M")} - {bitis.strftime("%H:%M")} '
                     f'({olay["sure_dakika"]:.1f}dk, {olay["anomali_sayisi"]} anomali, '
                     f'Max ΔT={olay["max_fark"]:.2f}°C, Trend: {olay["dominant_trend"]})',
                     fontsize=12, fontweight='bold')
            plt.ylabel('Sıcaklık (°C)')
            plt.grid(True, alpha=0.3)
            
            # X eksenini formatla
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            
            if i == len(gruplu_sonuclar) - 1:
                plt.xlabel('Zaman')
            
            plt.legend()
        
        plt.tight_layout()
        plt.show()

# ==================== 7. ANA FONKSİYON ====================
def main():
    """Ana çalışma fonksiyonu"""
    print("🏠 BUZDOLABI KAPI AÇILIMI ANALİZİ")
    print("=" * 50)
    
    # 1. Veriyi oku
    df = veri_oku()
    if df is None:
        return
    
    # 2. Kapak açılımı analizi yap
    anomali_detaylari = kapi_acilimi_analiz(df)
    
    # 3. Anomalileri grupla
    print("\n=== 3. ANOMALİ GRUPLAMA ===")
    gruplu_sonuclar = grupla_anomaliler(anomali_detaylari)
    
    print(f"\n📊 TOPLAM KAPI AÇILIMI OLAYI: {len(gruplu_sonuclar)}")
    
    # 4. Gruplanmış sonuçları göster
    if gruplu_sonuclar:
        print("\n📋 GRUPLANMIŞ KAPI AÇILIMI OLAYLARI:")
        print("=" * 50)
        
        for i, olay in enumerate(gruplu_sonuclar, 1):
            print(f"\n🚪 OLAY {i}:")
            print(f"   Başlangıç: {olay['baslangic']}")
            print(f"   Bitiş: {olay['bitis']}")
            print(f"   Süre: {olay['sure_dakika']:.1f} dakika")
            print(f"   Anomali Sayısı: {olay['anomali_sayisi']}")
            print(f"   Maksimum Fark: {olay['max_fark']:.2f}°C")
            print(f"   Dominant Trend: {olay['dominant_trend']}")
            
            # İsteğe bağlı: Detaylı anomalileri göster
            print("   Detaylar:")
            for j, detay in enumerate(olay['detaylar'], 1):
                print(f"     {j}. {detay['tarih'].strftime('%H:%M:%S')} | {detay['fark']:.2f}°C | {detay['trend']}")
    
    # 5. Grafikleri çiz
    grafigi_ciz(df, gruplu_sonuclar)

# ==================== PROGRAMI ÇALIŞTIR ====================
if __name__ == "__main__":
    main()