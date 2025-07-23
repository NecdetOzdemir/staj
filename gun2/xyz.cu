#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <cmath>

using namespace std;

// Yapılandırma parametreleri
const int NUM_TREES = 10000;         // Ağaç sayısını 1000'e çıkardık
const int SUBSAMPLE_SIZE = 2048;    // Örneklem boyutunu 1024'e çıkardık
const int BASE_MAX_DEPTH = 12;      // Temel maksimum derinlik
const int BLOCK_SIZE = 256;         // CUDA blok boyutu

struct Veri {
    int record_id;
    double temp;
};

// Rastgele sayı üreteci başlatma kernel'ı
__global__ void setup_kernel(curandState *state, unsigned long seed, int total_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_states) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// CUDA kernel for parallel depth calculation
__global__ void calculateDepths(const double* data, const double* targets, int* depths, 
                               int dataSize, int numTargets, int maxDepth, int subsampleSize, 
                               curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTargets) return;

    double target = targets[idx];
    int totalDepth = 0;

    // Dinamik maksimum derinlik hesaplama
    int dynamicMaxDepth = maxDepth + (int)(log2f((float)subsampleSize/256.0f));

    for (int tree = 0; tree < NUM_TREES; tree++) {
        curandState local_state = states[idx * NUM_TREES + tree];
        
        // Create subsample
        double currentSample[SUBSAMPLE_SIZE];
        for (int i = 0; i < subsampleSize; i++) {
            int randIndex = curand(&local_state) % dataSize;
            currentSample[i] = data[randIndex];
        }

        // Calculate isolation depth
        int depth = 0;
        int currentSize = subsampleSize;
        double workingSample[SUBSAMPLE_SIZE];
        for (int i = 0; i < subsampleSize; i++) workingSample[i] = currentSample[i];

        while (currentSize > 1 && depth < dynamicMaxDepth) {
            int randIndex = curand(&local_state) % currentSize;
            double splitValue = workingSample[randIndex];

            double left[SUBSAMPLE_SIZE], right[SUBSAMPLE_SIZE];
            int leftCount = 0, rightCount = 0;

            for (int i = 0; i < currentSize; i++) {
                if (workingSample[i] < splitValue) {
                    left[leftCount++] = workingSample[i];
                } else {
                    right[rightCount++] = workingSample[i];
                }
            }

            if (leftCount == 0 || rightCount == 0) break;

            if (target < splitValue) {
                currentSize = leftCount;
                for (int i = 0; i < leftCount; i++) workingSample[i] = left[i];
            } else {
                currentSize = rightCount;
                for (int i = 0; i < rightCount; i++) workingSample[i] = right[i];
            }

            depth++;
        }

        totalDepth += depth;
        states[idx * NUM_TREES + tree] = local_state;
    }

    depths[idx] = totalDepth / NUM_TREES;
}

int main() {
    srand(time(0));

    ifstream dosya("temp.csv");
    if (!dosya.is_open()) {
        cerr << "❌ Dosya acilamadi. 'temp.csv' kontrol et." << endl;
        return 1;
    }

    string satir;
    getline(dosya, satir); // Skip header

    vector<Veri> veriSeti;
    vector<double> sicaklikVerisi;

    // Read data from file
    while (getline(dosya, satir)) {
        stringstream ss(satir);
        string hucre;
        Veri veri;
        bool satirGecerli = true;

        if (!getline(ss, hucre, ',')) satirGecerli = false;
        else {
            try { veri.record_id = stoi(hucre); }
            catch (...) { satirGecerli = false; }
        }

        for (int i = 0; i < 3; i++) getline(ss, hucre, ',');

        if (!getline(ss, hucre, ',')) satirGecerli = false;
        else {
            if (hucre == "NA") satirGecerli = false;
            else {
                try { veri.temp = stod(hucre); }
                catch (...) { satirGecerli = false; }
            }
        }

        if (satirGecerli) {
            veriSeti.push_back(veri);
            sicaklikVerisi.push_back(veri.temp);
        }
    }
    dosya.close();

    cout << "✅ Toplam veri sayisi: " << veriSeti.size() << endl;
    cout << "✅ Sicaklik verisi sayisi: " << sicaklikVerisi.size() << endl;
    cout << "⚙️  Ağaç sayısı: " << NUM_TREES << endl;
    cout << "⚙️  Örneklem boyutu: " << SUBSAMPLE_SIZE << endl;

    int numTargets = veriSeti.size();
    
    // Dinamik maksimum derinlik hesaplama
    int dynamicMaxDepth = BASE_MAX_DEPTH + (int)(log2((double)SUBSAMPLE_SIZE/256.0));
    cout << "⚙️  Dinamik maksimum derinlik: " << dynamicMaxDepth << endl;

    // Prepare data for GPU
    thrust::host_vector<double> h_data = sicaklikVerisi;
    thrust::host_vector<double> h_targets(numTargets);
    for (int i = 0; i < numTargets; i++) {
        h_targets[i] = veriSeti[i].temp;
    }

    thrust::device_vector<double> d_data = h_data;
    thrust::device_vector<double> d_targets = h_targets;
    thrust::device_vector<int> d_depths(numTargets);

    // Initialize random states with curand
    int totalRandomStates = numTargets * NUM_TREES;
    thrust::device_vector<curandState> d_states(totalRandomStates);
    
    int setupBlocks = (totalRandomStates + BLOCK_SIZE - 1) / BLOCK_SIZE;
    setup_kernel<<<setupBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_states.data()), time(0), totalRandomStates);
    cudaDeviceSynchronize();

    // Calculate block and grid sizes
    int gridSize = (numTargets + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    calculateDepths<<<gridSize, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_targets.data()),
        thrust::raw_pointer_cast(d_depths.data()),
        h_data.size(),
        numTargets,
        BASE_MAX_DEPTH,
        SUBSAMPLE_SIZE,
        thrust::raw_pointer_cast(d_states.data())
    );

    cudaDeviceSynchronize();

    // Copy results back
    thrust::host_vector<int> h_depths = d_depths;

    // Anomali eşik değerini dinamik olarak hesapla
    double avg_depth = thrust::reduce(h_depths.begin(), h_depths.end()) / (double)numTargets;
    int anomaly_threshold = (int)(avg_depth * 0.6);  // Ortalamanın %60'ından düşük olanlar anomali

    cout << "\n📊 Anomali Tespiti (Eşik: " << anomaly_threshold << "):\n";
    for (size_t i = 0; i < veriSeti.size(); ++i) {
        if (h_depths[i] < 8) {
            cout << "ID: " << veriSeti[i].record_id
                 << " | Temp: " << veriSeti[i].temp
                 << " | Derinlik: " << h_depths[i];
            cout << " <-- ⚠️ ANOMALI!";
            cout << endl;
        }
    }

    return 0;
}