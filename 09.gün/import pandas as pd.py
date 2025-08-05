import pandas as pd
import numpy as np

# ==================== AYARLAR ====================
# Eşik değerleri (buradan değiştirilebilir)
ESIK_DUSEN = 0.10     # Düşen trendde kapı açılımı için eşik (°C)
ESIK_YUKSELEN = 0.20   # Yükselen trendde kapı açılımı için eşik (°C)

# CSV dosya yolu
CSV_PATH = "/home/necdet/Masaüstü/dolap/temp.csv"

# ==================== 1. VERİ OKUMA ====================
def veri_oku():
    """CSV dosyasını okur ve temizler"""
    print("=== 1. VERİ OKUMA ===")
    try:
        df = pd.read_csv(CSV_PATH)
        print("✅ CSV dosyası başarıyla okundu")
        
        # Tarih sütununu datetime formatına çevir
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        
        # Value sütununu sayısal formata çevir
        df['Value'] = pd.to_numeric(df['Value'])
        
        # Verileri sırala
        df_sorted = df.sort_values('Tarih').reset_index(drop=True)
        
        print(f"Toplam veri noktası: {len(df_sorted)}")
        print(f"Tarih aralığı: {df_sorted['Tarih'].min()} - {df_sorted['Tarih'].max()}")
        print(f"Sıcaklık aralığı: {df_sorted['Value'].min():.2f}°C - {df_sorted['Value'].max():.2f}°C")
        
        return df_sorted
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        return None

# ==================== 2. TREND ANALİZİ ====================
def trend_belirle(index, df_data, window=5):
    """Belirli bir noktanın trendini belirle"""
    if index < window:
        return "bilinmiyor"
    
    # Önceki window sayıda noktaya bak
    prev_values = df_data['Value'].iloc[index-window:index]
    if len(prev_values) < 2:
        return "bilinmiyor"
    
    try:
        trend_slope = np.polyfit(range(len(prev_values)), prev_values, 1)[0]
        
        if trend_slope < -0.005:  # Düşen trend
            return "dusen"
        elif trend_slope > 0.005:  # Yükselen trend
            return "yukselen"
        else:
            return "sabit"
    except:
        return "bilinmiyor"

# ==================== 3. KAPAK AÇILIMI ANALİZİ ====================
def kapi_acilimi_analiz(df_sorted):
    """Kapak açılımı tespiti yapar"""
    print("\n=== 3. KAPAK AÇILIMI ANALİZİ ===")
    
    # Zaman farkları (dakika cinsinden) ve sıcaklık farklarını hesapla
    time_diffs = df_sorted['Tarih'].diff().dt.total_seconds() / 60  # dakika cinsinden
    temp_diffs = df_sorted['Value'].diff()
    
    # Değişim hızı (°C/dakika)
    change_rates = temp_diffs / time_diffs
    change_rates.iloc[0] = 0  # İlk satır NaN olacağı için 0 yap
    
    # Yeni sütunları ekle
    df_sorted['change_rate'] = change_rates
    df_sorted['time_diff_min'] = time_diffs
    df_sorted['temp_diff'] = temp_diffs
    
    print(f"Eşik değerler: Düşen trend = {ESIK_DUSEN}°C, Yükselen trend = {ESIK_YUKSELEN}°C")
    
    # KAPAK AÇILIMI TESPİTİ
    kapi_acilimlar = []
    
    for i in range(1, len(df_sorted)):
        onceki_deger = df_sorted['Value'].iloc[i-1]
        simdiki_deger = df_sorted['Value'].iloc[i]
        fark = simdiki_deger - onceki_deger
        
        # Trendi belirle
        trend = trend_belirle(i-1, df_sorted)
        
        # DÜŞEN TRENDDE KAPAK AÇILIMI
        if trend == "dusen" and fark > ESIK_DUSEN:
            print(f"\n🔍 Düşen trendde ani artış tespit edildi:")
            print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
            print(f"   {onceki_deger:.2f}°C → {simdiki_deger:.2f}°C (fark: {fark:.2f}°C)")
            print(f"   → 🚪 KAPI AÇILDI")
            
            kapi_acilimlar.append({
                'tarih': df_sorted['Tarih'].iloc[i],
                'tip': 'kapi_acilimi',
                'aciklama': 'Düşen trendde ani artış',
                'fark': fark,
                'trend': 'dusen'
            })
        
        # YÜKSELEN TRENDDE KAPAK AÇILIMI
        elif trend == "yukselen" and fark > ESIK_YUKSELEN:
            print(f"\n🔍 Yükselen trendde ani artış tespit edildi:")
            print(f"   Tarih: {df_sorted['Tarih'].iloc[i]}")
            print(f"   {onceki_deger:.2f}°C → {simdiki_deger:.2f}°C (fark: {fark:.2f}°C)")
            print(f"   → 🚪 KAPI AÇILDI")
            
            kapi_acilimlar.append({
                'tarih': df_sorted['Tarih'].iloc[i],
                'tip': 'kapi_acilimi',
                'aciklama': 'Yükselen trendde ani artış',
                'fark': fark,
                'trend': 'yukselen'
            })
    
    print(f"\n📊 TOPLAM KAPI AÇILIMI: {len(kapi_acilimlar)}")
    
    return kapi_acilimlar

# ==================== 4. ANA FONKSİYON ====================
def main():
    """Ana çalışma fonksiyonu"""
    print("🏠 DOLAP SICAKLIK ANALİZİ")
    print("=" * 50)
    
    # 1. Veriyi oku
    df = veri_oku()
    if df is None:
        return
    
    # 2. Kapak açılımı analizi yap
    sonuclar = kapi_acilimi_analiz(df)
    
    # 3. Detaylı sonuçları göster
    if sonuclar:
        print("\n📋 DETAYLI SONUÇLAR:")
        print("=" * 50)
        for i, olay in enumerate(sonuclar, 1):
            print(f"{i:2d}. {olay['tarih']} | {olay['tip']} | Fark: {olay['fark']:.2f}°C")
            print(f"    Trend: {olay['trend']} | {olay['aciklama']}")

# ==================== PROGRAMI ÇALIŞTIR ====================
if __name__ == "__main__":
    main()