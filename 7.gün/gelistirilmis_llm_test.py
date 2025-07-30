import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import random

# ----------------- AYARLAR -----------------
TEST_CSV_PATH = "/home/necdet/Masaüstü/staj-main/7.gün/test.csv"
TRAIN_CSV_PATH = "/home/necdet/Masaüstü/staj-main/7.gün/weather.csv"
MODEL_FOLDER = "/home/necdet/Masaüstü/staj-main/7.gün/egitim_dosyalari"
SCALER_PATH = os.path.join(MODEL_FOLDER, "weather_scalers.save")
THRESHOLD_RATIO = 0.2  # Anomali olarak kabul edilen hata eşiği
FEATURE_NAME = "Sicaklik"
SAAT_DILIMI = 2  # Saat dilimi genişliği

# ------------- CİHAZ AYARI (CPU/GPU) --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ---------------- AUTOENCODER MODEL TANIMI ----------------
class Autoencoder(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
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
            nn.Linear(16, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ----------------- ZAMAN DİLİMİ DÖNÜŞTÜRÜCÜ -----------------
def saat_to_dilim(saat):
    hour = int(saat.split(':')[0])
    start = (hour // SAAT_DILIMI) * SAAT_DILIMI
    return f"{start:02d}-{start+SAAT_DILIMI-1:02d}"

# ---------------- BENZER DOKÜMANLARI SEÇME FONKSİYONU ----------------
def retrieve_similar_docs(query, documents, doc_embeddings, model, top_k=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    return [documents[i] for i in top_k_indices]

# ---------------- PROMPT OLUŞTURUCU ----------------
def olustur_prompt_rag(value, prediction, diff, timestamp, same_dilim_samples, max_min_samples, near_dilim_samples):
    context = "\n".join([
        "Samples from same time slot:",
        *same_dilim_samples,
        "Max and Min values from same slot:",
        *max_min_samples,
        "Samples from nearby time slots:",
        *near_dilim_samples
    ])

    prompt = f"""
You are a highly experienced meteorologist and data scientist with 20 years of experience analyzing temperature data and detecting anomalies.

Consider the historical temperature data samples provided from the same and nearby time slots.

Analyze the suspicious data point below step-by-step:

- Is this data point an anomaly? Answer clearly with 'Yes' or 'No'.
- Explain your reasoning in detail, including statistical and meteorological considerations.
- If it is an anomaly, speculate briefly on possible causes (measurement error, sudden weather change, etc.).
- Consider if this could be natural variation or an error.

Data Point:
Timestamp: {timestamp}
Actual temperature: {value:.2f} °C
Predicted temperature: {prediction:.2f} °C
Difference: {diff:.2f} °C

Relevant historical data:
{context}

Please structure your response as:
1. Anomaly status (Yes/No)
2. Detailed explanation
3. Possible causes (if anomaly)
"""
    return prompt


# ---------------- LLM SORGULAMA FONKSİYONU ----------------
def deepseek_sorgula(prompt):
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1', prompt],
            capture_output=True,
            text=True,
            timeout=2000
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error occurred: {e}"

# ----------------- TEST VE ANOMALİ ANALİZİ ------------------
def test_and_analyze():
    # Test verisini oku ve zaman dilimi ekle
    df_test = pd.read_csv(TEST_CSV_PATH)
    df_test.columns = df_test.columns.str.strip()
    df_test['ZamanDilimi'] = df_test['Saat'].apply(saat_to_dilim)

    # Eğitim verisini oku ve zaman dilimi ekle
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_train.columns = df_train.columns.str.strip()
    df_train['ZamanDilimi'] = df_train['Saat'].apply(saat_to_dilim)

    # Önceden kaydedilmiş scaler dictionary'sini yükle
    scaler_dict = joblib.load(SCALER_PATH)

    # SentenceTransformer modeli yükle
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # embedding model bazen to(device) desteklemez, sorun olursa yorum satırı yapabilirsin
    # embedding_model.to(device)

    # Eğitim verisinden metin dokümanları hazırla
    documents = [
        f"Day {row.get('Gün', 'NA')}, Hour {row['Saat']}: Sicaklik recorded was {row[FEATURE_NAME]:.2f}."
        for _, row in df_train.iterrows()
    ]
    doc_embeddings = embedding_model.encode(documents)

    # Her zaman dilimi için eğitimli modeli yükle
    models = {}
    for dilim in scaler_dict.keys():
        model_path = os.path.join(MODEL_FOLDER, f"autoencoder_{dilim}.pth")
        if os.path.exists(model_path):
            model = Autoencoder(input_size=1).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models[dilim] = model
        else:
            print(f"Model bulunamadı: {model_path}")

    # Test verisini dolaş ve anomali skorlarını hesapla
    real_values = []
    predicted_values = []
    error_ratios = []
    timestamps = []
    anomaly_indices = []

    for idx, row in df_test.iterrows():
        dilim = row['ZamanDilimi']
        if dilim not in scaler_dict or dilim not in models:
            continue  # Eğer ilgili model veya scaler yoksa atla

        scaler = scaler_dict[dilim]
        model = models[dilim]

        value = row[FEATURE_NAME]
        norm_val = scaler.transform([[value]])
        input_tensor = torch.tensor(norm_val, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()

        pred_val = scaler.inverse_transform(pred_norm)[0][0]
        error_ratio = abs(value - pred_val) / max(abs(value), 1e-6)

        # Sonuçları listeye ekle
        real_values.append(value)
        predicted_values.append(pred_val)
        error_ratios.append(error_ratio)
        timestamps.append(f"{row.get('Gün', 'NA')} {row['Saat']}")

        if error_ratio > THRESHOLD_RATIO:
            anomaly_indices.append(idx)

    # Sonuçları grafikle göster
    plt.figure(figsize=(16,6))
    plt.plot(timestamps, real_values, label='Actual', color='blue')
    plt.plot(timestamps, predicted_values, label='Predicted', linestyle='--', color='green')
    for i in anomaly_indices:
        plt.scatter(timestamps[i], real_values[i], color='red', label='Anomaly' if i == anomaly_indices[0] else "")
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Time")
    plt.ylabel(FEATURE_NAME)
    plt.title("Autoencoder-Based Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Anomaliler için LLM sorgusu
    print("\n🔍 LLM-Based Anomaly Analysis (llama3.1):")
    for idx in anomaly_indices:
        val = real_values[idx]
        pred = predicted_values[idx]
        diff = abs(val - pred)
        zaman = timestamps[idx]
        dilim = df_test.loc[idx, 'ZamanDilimi']

        # Aynı dilimden 5 rastgele örnek al
        same_dilim_samples_df = df_train[df_train['ZamanDilimi'] == dilim]
        same_dilim_samples = random.sample(
            [
                f"Day {row.get('Gün', 'NA')}, Hour {row['Saat']}: Sicaklik {row[FEATURE_NAME]:.2f}"
                for _, row in same_dilim_samples_df.iterrows()
            ],
            min(5, len(same_dilim_samples_df))
        )

        # Aynı dilimden max ve min değerleri al
        max_val = same_dilim_samples_df[FEATURE_NAME].max()
        min_val = same_dilim_samples_df[FEATURE_NAME].min()
        max_min_samples = [
            f"Max value: {max_val:.2f}",
            f"Min value: {min_val:.2f}"
        ]

        # Yakın dilimlerden 3 rastgele örnek al
        def get_nearby_dilims(dilim):
            start = int(dilim.split('-')[0])
            dilims = []
            for offset in [-SAAT_DILIMI, SAAT_DILIMI]:
                near_start = start + offset
                if 0 <= near_start <= 22:  # Gün içinde 0-23 saat arası
                    dilims.append(f"{near_start:02d}-{near_start+SAAT_DILIMI-1:02d}")
            return dilims

        nearby_dilims = get_nearby_dilims(dilim)
        near_dilim_samples = []
        for nd in nearby_dilims:
            near_df = df_train[df_train['ZamanDilimi'] == nd]
            if not near_df.empty:
                near_samples = random.sample(
                    [
                        f"Day {row.get('Gün', 'NA')}, Hour {row['Saat']}: Sicaklik {row[FEATURE_NAME]:.2f}"
                        for _, row in near_df.iterrows()
                    ],
                    min(2, len(near_df))
                )
                near_dilim_samples.extend(near_samples)

        # Promptu oluştur
        prompt = olustur_prompt_rag(val, pred, diff, zaman, same_dilim_samples, max_min_samples, near_dilim_samples)

        # LLM'ye sor
        yanit = deepseek_sorgula(prompt)

        print(f"\n🕒 {zaman} - Prompt:\n{prompt}\n\n🤖 Response:\n{yanit}\n{'-'*60}")

# ------------- PROGRAM ÇALIŞTIRMA ----------------
if __name__ == "__main__":
    test_and_analyze()
