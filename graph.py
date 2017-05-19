# grapher.py
import numpy as np
import matplotlib.pyplot as plt

with open("celebA_lost.txt") as loss:
	d_loss = []
	g_loss = []
	for line in loss:
		l = line.strip('\n').split(',')
		d = float(l[0])
		g = float(l[1])
		d_loss.append(d)
		g_loss.append(g)

d_loss = d_loss[0:-1:100]
g_loss = g_loss[0:-1:100]
x = range(len(d_loss))

plt.figure(1)
plt.subplot(211)
plt.plot(x, d_loss, 'r')
plt.xlabel("100s of iterations")
plt.ylabel("loss")
plt.axis([-10, 800, 0, 8])
plt.title('Discriminator Loss for T=1, MNIST')
plt.subplot(212)
plt.plot(x, g_loss, 'g')
plt.xlabel("100s of iterations")
plt.ylabel("loss")
plt.axis([-10, 800, 3, 15])
plt.title('Generator Loss for T=1, MNIST')
plt.show()