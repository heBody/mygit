
# 激活函数的理解
def xor(v1, v2):
    v3 = v1 + v2
    # 可以认为这是一个激活函数
    if v3 > 1.5:
        v3 = 1
    else:
        v3 = 0

    ret = v1 + v2 + v3 * (-2)
    # 可以认为这是一个激活函数
    if ret > 0.5:
        return 1
    else:
        return 0


print(xor(0, 0))
print(xor(1, 1))
print(xor(1, 0))
print(xor(0, 1))