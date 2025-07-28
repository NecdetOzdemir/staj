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

# ------------------ AYARLAR ------------------
TEST_CSV_PATH = "/home/necdet/Masaüstü/Yeni Klasör/egitim_dosyalari/test.csv"
TRAIN_CSV_PATH = "/home/necdet/Masaüstü/Yeni Klasör/egitim_dosyalari/weather.csv"
MODEL_PATH = "/home/necdet/Masaüstü/Yeni Klasör/egitim_dosyalari/weather_autoencoder.pth"
SCALER_PATH = "/home/necdet/Masaüstü/Yeni Klasör/egitim_dosyalari/weather_scalers.save"
THRESHOLD_RATIO = 0.2
FEATURE_NAME = "Sicaklik"
saat_dilimi = 2

# ------------------ CİHAZ AYARI (CPU/GPU) ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ------------------ ZAMAN DİLİMİ DÖNÜŞTÜRÜCÜ ------------------
def saat_to_dilim(saat):
    """Saat stringini alır, 2 saatlik zaman dilimine dönüştürür."""
    hour = int(saat.split(':')[0])
    start = (hour // saat_dilimi) * saat_dilimi
    return f"{start:02d}-{start+1:02d}"

# ------------------ AUTOENCODER MODELİ TANIMI ------------------
class Autoencoder(nn.Module):
    def __init__(self, input_size=1):
        """Basit 1 boyutlu girişli autoencoder modeli oluşturur."""
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
        """İleri besleme işlemi."""
        return self.decoder(self.encoder(x))

# ------------------ BENZER DOKÜMANLARI GETİRME FONKSİYONU ------------------
def retrieve_similar_docs(query, documents, doc_embeddings, model, top_k=5):
    """Sorgu ile en benzer 'top_k' dokümanı bulur."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    return [documents[i] for i in top_k_indices]

# ------------------ RAG İÇİN PROMPT OLUŞTURUCU ------------------
def olustur_prompt_rag(value, prediction, diff, timestamp, relevant_docs):
    """Anomali sorgusu için LLM prompt'u hazırlar."""
    context = "\n".join(relevant_docs)
    return (
        f"Here are the most relevant historical records:\n{context}\n\n"
        f"Suspicious data point details:\n"
        f"Timestamp: {timestamp}\n"
        f"Actual value = {value:.2f}\n"
        f"Predicted value = {prediction:.2f}\n"
        f"Difference = {diff:.2f}\n\n"
        "Is this value an anomaly? Please answer clearly with 'Yes' or 'No', "
        "and briefly explain your reasoning."
    )

# ------------------ LLM SORGULAMA (OLLAMA - llama3.1) ------------------
def deepseek_sorgula(prompt):
    """OLLAMA komutu ile prompt'u llama3.1 modeline gönderir, cevabı alır."""
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

# ------------------ TEST VE ANOMALİ ANALİZİ ------------------
def test_autoencoder_ve_deepseek():
    """
    Test verisi üzerinde autoencoder ile tahmin yapar,
    anomali tespiti yapar ve
    anormal noktaları LLM ile sorgular.
    """
    # Test ve eğitim verisini yükle, zaman dilimini hesapla
    df_test = pd.read_csv(TEST_CSV_PATH)
    df_test.columns = df_test.columns.str.strip()
    df_test['ZamanDilimi'] = df_test['Saat'].apply(saat_to_dilim)

    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_train.columns = df_train.columns.str.strip()

    input_dim = 1

    # Modeli oluştur ve yükle
    model_ae = Autoencoder(input_size=input_dim).to(device)
    model_ae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_ae.eval()

    # Scaler'ları yükle
    scaler_dict = joblib.load(SCALER_PATH)

    # Embedding modeli ve doküman hazırlama
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model.to(device)

    documents = [
        f"Day {row.get('Gün', 'NA')}, Hour {row['Saat']}: {FEATURE_NAME} recorded was {row[FEATURE_NAME]:.2f}."
        for _, row in df_train.iterrows()
    ]
    doc_embeddings = embedding_model.encode(documents)

    # Değişkenleri oluştur
    real_values, predicted_values, timestamps, anomaly_indices = [], [], [], []

    # Test verisi üzerinde döngü
    for i, row in df_test.iterrows():
        dilim = row['ZamanDilimi']
        value = row[FEATURE_NAME]
        saat = row['Saat']
        gun = str(row.get('Gün', i // 24 + 1))
        label = f"{gun} {saat}"

        if dilim not in scaler_dict:
            print(f"⚠️ Skipping {label}, scaler bulunamadı: {dilim}")
            continue

        scaler = scaler_dict[dilim]
        value_norm = scaler.transform([[value]])
        input_tensor = torch.tensor(value_norm, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed = model_ae(input_tensor).cpu().numpy()
        predicted = scaler.inverse_transform(reconstructed)[0][0]
        error_ratio = abs(value - predicted) / max(abs(value), 1e-6)

        real_values.append(value)
        predicted_values.append(predicted)
        timestamps.append(label)

        if error_ratio > THRESHOLD_RATIO:
            anomaly_indices.append(i)

    # Grafik çizimi
    plt.figure(figsize=(16, 6))
    plt.plot(timestamps, real_values, color='blue', label='Actual')
    plt.plot(timestamps, predicted_values, color='green', linestyle='--', label='Predicted')
    for idx in anomaly_indices:
        plt.scatter(timestamps[idx], real_values[idx], color='red', label='Anomaly' if idx == anomaly_indices[0] else "")
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Time")
    plt.ylabel(FEATURE_NAME)
    plt.title("Autoencoder-Based Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # LLM ile anomali analizini yap
    print("\n🔍 LLM-Based Anomaly Analysis (llama3.1):")
    for idx in anomaly_indices:
        val = real_values[idx]
        pred = predicted_values[idx]
        diff = abs(val - pred)
        zaman = timestamps[idx]

        query = f"{FEATURE_NAME} anomaly at {zaman}, actual={val:.2f}"
        relevant_docs = retrieve_similar_docs(query, documents, doc_embeddings, embedding_model)
        prompt = olustur_prompt_rag(val, pred, diff, zaman, relevant_docs)
        yanit = deepseek_sorgula(prompt)

        print(f"\n🕒 {zaman} - Prompt:\n{prompt}\n\n🤖 Response:\n{yanit}\n{'-'*60}")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    test_autoencoder_ve_deepseek()
