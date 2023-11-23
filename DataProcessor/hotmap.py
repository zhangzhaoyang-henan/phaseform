import numpy as np
import matplotlib.pyplot as plt 

# ax = np.linspace(-0.3, 0.3, 100).tolist()
# print(ax)

with open("tewD.csv") as rf:
    points = [line.split(',') for line in rf.readlines()]
points = np.array(points, dtype=np.float)
print(points.shape)
cmaps = [plt.cm.jet, plt.cm.gray, plt.cm.cool, plt.cm.hot]
plt.subplot(121)
plt.imshow(points, cmap=cmaps[0])
x2d = 49.5
y2d = 49.384
x1d = 49.5
y1d = 51.975
plt.plot(x2d,y2d, 'bo', c="black", ms=10)
plt.plot(x1d, y1d, 'bo', c="white", ms=5)
plt.annotate('(1D search Idx)', xy=(x1d, y1d), xytext=(x1d+2,y1d+3)) 
plt.annotate('(2D search Idx)', xy=(x2d,y2d), xytext=(x2d-2,y2d-3)) 
plt.xlabel("a3 idx")
plt.ylabel("a2 idx")
plt.subplot(122)
plt.imshow(points, cmap=cmaps[0])
plt.show()