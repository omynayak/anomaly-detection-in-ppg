import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import streamlit as st

st.title("Isolation forest plot")
fileNo = st.number_input("Enter the file number(0-22) : ",min_value=0, max_value=22, value=0)
fileName = f"final_{fileNo}_ppg"

estimator = st.number_input("Enter the number of trees: ",min_value=100, max_value=600, value=100)
cont = st.number_input("Enter the contamination: ", min_value=0.001, max_value=1.00)

data = []
timeStamps = []
with open(f"Data/LMS_Median/{fileName}.csv", "r") as file:
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
print(features.shape)  


estimator = 500
cont = 0.0125
rs = 42
clf = IsolationForest(
    n_estimators=estimator,
    contamination=cont,  
    random_state=rs
)
clf.fit(features)
preds = clf.predict(features)   
scores = clf.decision_function(features)

import matplotlib.pyplot as plt

anomaly_indices = np.where(preds == -1)[0]
anomaly_times = [timeStamps[i * stride] for i in anomaly_indices]

fig,ax = plt.subplots()
ax.scatter(anomaly_times, ppg_norm[anomaly_indices * stride], color='r', label="Anomalies")
ax.plot(timeStamps, ppg_norm, label="Normalized PPG", alpha=0.7)
ax.legend()
ax.set_xlabel("Time(ms)")
ax.set_ylabel("PPG")
ax.set_title(f"Total Trees: {estimator}, Contamination: {cont}")
ax.grid()

fig2,ax2 = plt.subplots()
ax2.plot(timeStamps, data)
ax2.legend()
ax2.set_xlabel("Time(ms)")
ax2.set_ylabel("PPG")
ax2.set_title("Raw PPG Data")
ax2.grid()


st.pyplot(fig)
st.pyplot(fig2)
