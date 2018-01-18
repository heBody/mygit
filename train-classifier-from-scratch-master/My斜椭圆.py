import math
import matplotlib.pyplot as plt

# 绘出椭圆来
a = 5  # x轴
b = 12  # y轴
angle = -math.pi / 6  # 角度
plt.ion()
for j in range(1000):
    angle += 0.03
    plt.clf()
    t = -math.pi
    while t <= math.pi:
        t += 0.05
        y = a*math.cos(t) * math.sin(angle) + b * math.sin(t) * math.cos(angle)
        x = a*math.cos(t) * math.cos(angle) - b * math.sin(t) * math.sin(angle)
        plt.scatter(x, y)

    plt.pause(0.05)
plt.show()
