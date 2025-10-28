import csv
import matplotlib.pyplot as plt
import numpy as np

file_name = "final_14_ppg.csv"

timeStamps = []
data  = []

with open(file_name, "r") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        timeStamps.append(int(row[0]))
        data.append(float(row[1]))

plt.plot(timeStamps, data)
plt.show()