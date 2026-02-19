import serial
import struct
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np



ser = serial.Serial("COM4", 1000000)

ser.reset_input_buffer()

#print(data[:10])
#print("Mean:", data.astype(np.uint32).mean())


max_points = 500000
data = deque([0.0] * max_points, maxlen=max_points)    #   A fixed-size rolling buffer pre-filled with zeros that automatically keeps only the most recent max_points values — perfect for real-time plots.

start = time.time()
now = time.time()



while (now-start)<40: #True:
    raw = ser.read(2)
    value = struct.unpack('<H', raw)[0]  # unsigned 16-bit
    #print(value)  # 0–4095
    data.append(value)


    #raw = ser.read(4096)
    #datain = np.frombuffer(raw, dtype=np.uint16)
    #data.append(datain)
    
    # if 1<(now-start)<1.005:
    #     print(now-start)
    now = time.time()


fig, ax = plt.subplots()
line, = ax.plot(data, linestyle='None', marker='o')
#ax.set_ylim(-2000, 2000)   # adjust to your data range
ax.set_xlim(0, 500_000)   # adjust to your data range
ax.set_title(" Serial Data minus mean")
ax.set_xlabel("Sample index")
ax.set_ylabel("Value")    
plt.savefig("colloid_data_3.png")
plt.show()

#calculate autocorrelation function of the data
#acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
#acf = acf[acf.size // 2:] / acf[acf.size // 2]

def autocorr_limited(x, max_lag):
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)

    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[:N-lag], x[lag:])

    return acf / acf[0]

acf = autocorr_limited(list(data)[250000:], 100000)
fig, ax = plt.subplots()
line, = ax.plot(acf, linestyle='None', marker='o')
ax.set_ylim(-1, 1)   # adjust to your data range
ax.set_xlim(0, 100000)   # adjust to your data range
ax.set_title("Autocorrelation of (Data- mean)")
ax.set_xlabel("Sample interval")
ax.set_ylabel("Value")
plt.savefig("dls-colloid_3.png")    
plt.show()