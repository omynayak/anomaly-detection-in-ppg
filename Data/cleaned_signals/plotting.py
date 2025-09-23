import matplotlib.pyplot as plt
import csv
import numpy as np

file_name = "cleaned_P12_ppg.csv"

timeStamps = []
ppgData = []
with open(file_name, mode="r") as file:
    data = csv.reader(file)
    next(data)  

    for row in data:
        timeStamps.append(int(row[0]))
        ppgData.append(float(row[1]))

timeStamps = np.array(timeStamps)
ppgData = np.array(ppgData)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(timeStamps, ppgData)
plt.grid()
plt.title(f"Cleaned PPG Signal (Time Domain)\n{file_name}")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")


dt = np.median(np.diff(timeStamps)) / 1000.0  
fs = 1.0 / dt

n = len(ppgData)
freqs = np.fft.rfftfreq(n, d=dt)
fft_vals = np.abs(np.fft.rfft(ppgData))

plt.subplot(1, 2, 2)
plt.plot(freqs, fft_vals)
plt.grid()
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
