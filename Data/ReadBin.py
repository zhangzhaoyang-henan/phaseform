import os
import struct
import matplotlib.pyplot as plt
import matplotlib


with open('D:/Data/wav') as rf:
	line = rf.readlines()[0]
lines = [int(nums) for nums in line.split(',')]

count = 0
for idx, l in enumerate(lines):
	if idx == len(lines)-1:
		break
	if lines[idx + 1]- lines[idx ]> 4000:
		count += 1
print(count)

# simhei.ttf

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

# f = open("D:/cervical-cancer/沈阳/激光问题/B1821232-1_circle_5.0x5.0_C01_S0004.bin", "rb")
f = open("D:/Data/2019/M0008_2019_P0000013/M0008_2019_P0000013_3D_2.0x2.0_C03_S004.bin", "rb")

data = []
f.seek(2048*400*320*2)
for i in range(2048):
    re = f.read(2)
    data.append(struct.unpack("H", re)[0])
plt.plot(range(2048), data)

plt.show()