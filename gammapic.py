import numpy as np
import matplotlib.pyplot as plt


x = [0.0001, 0.001, 0.01, 0.1, 1]
y1 = [0.8461, 0.8646, 0.8813, 0.8801, 0.8700]
y2 = [0.7490, 0.7709, 0.8061, 0.8661, 0.8611]

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.plot(x, y1, 'r', label='ACC')
ax1.set_xlabel('$\gamma$')
ax1.set_ylabel('ACC')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'g', label='NMI')
ax2.set_ylabel('NMI')

plt.savefig('./gamma.jpg')