import os
import struct

# f = open("D:/cervical-cancer/沈阳/激光问题/B1821232-1_circle_5.0x5.0_C01_S0004.bin", "rb")
f = open("D:/Data/2019/M0008_2019_P0000067/M0008_2019_P0000067_circle_2.0x2.0_C05_S002.tiff", "rb")

data = []
f.seek(8)
# re = f.read(2)
# print('gender:{}'.format(struct.unpack("H", re)[0]))
# re = f.read(2)
# print('age:{}'.format(struct.unpack("H", re)[0]))
# re = f.read(4)
# print('hpv:{}'.format(struct.unpack("i", re)[0]))
# re = f.read(4)
# print('tct:{}'.format(struct.unpack("i", re)[0]))
re = f.read(4)
print('x pixel size:{}'.format(struct.unpack("f", re)[0]))
re = f.read(4)
print('y pixel size:{}'.format(struct.unpack("f", re)[0]))
# data.append(struct.unpack("H", re)[0])