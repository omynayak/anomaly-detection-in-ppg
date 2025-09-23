import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import csv

def lms_filter(accel_signal, ppg_signal, mu=1e-5, filter_order=32):
    n_samples = len(ppg_signal)
    weights = np.zeros(filter_order)
    motion_approx = np.zeros(n_samples)
    cleaned_ppg = np.zeros(n_samples)

    for i in range(filter_order, n_samples):
        x = accel_signal[i-filter_order:i][::-1]   
        y = np.dot(weights, x)                     
        error = ppg_signal[i] - y                  
        weights = weights + 2 * mu * error * x     
        motion_approx[i] = y
        cleaned_ppg[i] = error                     

    return cleaned_ppg, motion_approx, weights


timeStamps = []
ppgData = []
with open("PPG.csv") as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        timeStamps.append(int(row[1]))
        ppgData.append(float(row[2]))

timeStamps = np.array(timeStamps)
ppgData = np.array(ppgData)


accX = []
accY = []
accZ = []
accData = []
with open("ACC.csv", mode="r") as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        tempX = float(row[2])
        tempY = float(row[3])
        tempZ = float(row[4])
        accX.append(tempX)
        accY.append(tempY)
        accZ.append(tempZ)
        mag = (tempX**2 + tempY**2 + tempZ**2) ** 0.5
        accData.append(mag)

accData = np.array(accData)


cleaned_signal, motion_approx, weights = lms_filter(accData, ppgData, mu=1e-5, filter_order=32)



def butter_lowpass_filter(signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        raise ValueError(f"Cutoff {cutoff} Hz must be less than Nyquist {nyquist} Hz. "
                         f"Check your timeStamps units or sampling rate.")
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


fs = 1000 / np.mean(np.diff(timeStamps))   
cutoff = 3

cleaned_lp = butter_lowpass_filter(cleaned_signal, cutoff, fs)


plt.figure(figsize=(12,5))
plt.plot(timeStamps, cleaned_lp, label="LMS + Lowpass (2.5 Hz)")
plt.legend()
plt.title("LMS Lowpass Filtered")


plt.figure(figsize=(12,5))
plt.subplot(2,2,1)
plt.plot(timeStamps, accX, label="X axis")
plt.legend()
plt.title("X axis Accelerometer")
plt.grid()

plt.subplot(2,2,2)
plt.plot(timeStamps, accY, label="Y axis")
plt.legend()
plt.title("Y axis Accelerometer")

plt.grid()
plt.subplot(2,2,3)
plt.plot(timeStamps, accZ, label="Z axis")
plt.legend()
plt.title("Z axis Accelerometer")

plt.grid()
plt.subplot(2,2,4)
plt.plot(timeStamps, ppgData, label="PPG")
plt.legend()
plt.title("PPG Data")
plt.grid()


plt.figure(figsize=(12,5))
plt.plot(timeStamps, ppgData, label="Noisy PPG")
plt.plot(timeStamps, cleaned_signal, label="Cleaned PPG")
plt.legend()
plt.title("PPG before vs after LMS")
plt.grid()



plt.tight_layout()
plt.show()
