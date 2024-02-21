import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
import csv

csvfile = open('./pretrain_log1.csv', 'r')
plots = csv.reader(csvfile, delimiter=',')
x=[]
y=[]

for row in plots:
    x.append((row[0]))
    y.append((row[1]))

x = list(map(int, x))
y = list(map(float, y))
print(x, y)


plt.plot(x, y)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('mdecpreloss.png')
