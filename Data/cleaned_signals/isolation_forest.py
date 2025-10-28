import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

fileName = "cleaned_P08_ppg"

data = []
timeStamps = []
with open(f"{fileName}.csv", "r") as file:
    stuff = csv.reader(file)

    next(stuff)
    for row in stuff:
        timeStamps.append(int(row[0]))
        data.append(float(row[1]))

data = np.array(data)
timeStamps = np.array(timeStamps)

scaler = StandardScaler()
ppg_norm = scaler.fit_transform(data.reshape(-1, 1)).flatten()

fs = 1000 / np.mean(np.diff(timeStamps))  
print(f"Sampling frequency ≈ {fs:.2f} Hz")

window_sec = 2       
stride_sec = 1       
window_size = int(window_sec * fs)
stride = int(stride_sec * fs)

segments = []
for start in range(0, len(ppg_norm) - window_size, stride):
    segment = ppg_norm[start:start + window_size]
    segments.append(segment)
segments = np.array(segments)
print("Total segments:", len(segments))

def extract_features(seg, fs):
    mean = np.mean(seg)
    std = np.std(seg)
    skew = np.mean(((seg - mean) / std)**3)
    kurt = np.mean(((seg - mean) / std)**4) - 3
    energy = np.sum(seg**2)
    dom_freq = np.abs(np.fft.rfft(seg))
    freq_bins = np.fft.rfftfreq(len(seg), 1/fs)
    dominant_frequency = freq_bins[np.argmax(dom_freq)]
    return [mean, std, skew, kurt, energy, dominant_frequency]

features = np.array([extract_features(s, fs) for s in segments])
print(features.shape)  # (n_segments, n_features)



clf = IsolationForest(
    n_estimators=500,
    contamination=0.025,  
    random_state=42
)
clf.fit(features)
preds = clf.predict(features)   
scores = clf.decision_function(features)

import matplotlib.pyplot as plt

anomaly_indices = np.where(preds == -1)[0]
anomaly_times = [timeStamps[i * stride] for i in anomaly_indices]

plt.figure(figsize=(14,4))
plt.plot(timeStamps, ppg_norm, label="Normalized PPG", alpha=0.7)
plt.scatter(anomaly_times, ppg_norm[anomaly_indices * stride], color='r', label="Anomalies")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.legend()
plt.title("PPG Anomaly Detection using Isolation Forest")
plt.grid()
plt.show()