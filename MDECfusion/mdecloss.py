import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
import csv

csvfile = open('./mdec_log1.csv', 'r')
plots = csv.reader(csvfile, delimiter=',')
x=[]
y=[]
# l = []
l_c = []
l1 = []
l2 = []
l3 = []
for row in plots:
    y.append((row[1]))
    l_c.append((row[5]))
    l1.append((row[6]))
    l2.append((row[7]))
    l3.append((row[8]))
    x.append((row[0]))

x = list(map(int, x))
y = list(map(float, y))
# l = list(map(float, l))
l_c = list(map(float, l_c))
l1 = list(map(float, l1))
l2 = list(map(float, l2))
l3 = list(map(float, l3))
print(x, y, l_c, l1, l2, l3)
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x,y)
ax1.plot(x, y, 'r', label='ACC')
ax1.set_ylabel('ACC')
ax1.set_xlabel('iter')

ax2 = ax1.twinx()

ax2.plot(x, l_c, 'g', label='lc')
ax2.plot(x, l1, 'b', label='l1')
ax2.plot(x, l2, 'y', label='l2')
ax2.plot(x, l3, 'c', label='l3')
# ax2.plot(x, l, 'b', label='l')
ax2.set_ylabel('loss')

# plt.xlabel('iter')


plt.savefig('./mdecloss.png')




