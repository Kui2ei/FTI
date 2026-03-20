import cmath
import math

def fft_recursive(x):
    """
    递归版 radix-2 Cooley–Tukey FFT
    :param x: 可迭代对象，长度为 2 的幂，元素为复数（或可转为复数）
    :return: 长度相同的列表，对应频域结果
    """
    n = len(x)
    if n == 1:
        return list(x)
    # 检查是否为 2 的幂
    if n & (n - 1) != 0:
        raise ValueError("输入长度必须是 2 的幂")

    # 偶数下标与奇数下标拆分
    even = fft_recursive(x[0::2])
    odd  = fft_recursive(x[1::2])

    result = [0] * n
    for k in range(n // 2):
        # 旋转因子 twiddle factor: e^{-2πik/n}
        twiddle = cmath.exp(-2j * math.pi * k / n) * odd[k]
        result[k] = even[k] + twiddle
        result[k + n // 2] = even[k] - twiddle
    return result




# 迭代版fft

def _bit_reverse_indices(n: int):
    """
    计算 0..n-1 的 bit-reversal 映射（n 为 2 的幂）。
    返回数组 rev，使得新序列[i] = 原序列[rev[i]]
    """
    bits = n.bit_length() - 1  # log2(n)
    rev = [0] * n
    for i in range(n):
        x = i
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev

def fft_iterative(x):
    """
    迭代版 radix-2 FFT（DIT，含 bit-reversal 置换 + 蝶形运算）
    :param x: 可迭代对象，长度为 2 的幂
    :return: FFT 结果列表
    """
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError("输入长度必须是 2 的幂")

    # 拷贝一份作为工作数组
    a = list(x)

    # 1) bit-reversal 重排
    rev = _bit_reverse_indices(n)
    a = [a[rev[i]] for i in range(n)]

    # 2) 逐层蝶形运算
    m = 2
    while m <= n:
        angle = -2 * math.pi / m
        wm = complex(math.cos(angle), math.sin(angle))  # 每一层公用的旋转因子
        half = m // 2
        for k in range(0, n, m):
            w = 1 + 0j
            for j in range(half):
                t = w * a[k + j + half]
                u = a[k + j]
                a[k + j] = u + t           # 上半部分
                a[k + j + half] = u - t    # 下半部分
                w *= wm                    # 乘上下一步的旋转因子
        m <<= 1  # m *= 2
    return a

if __name__ == "__main__":
    data = [1, 2, 3, 4, 0, 0, 0, 0]  # 长度必须为 2 的幂

    fr = fft_recursive(data)
    fi = fft_iterative(data)

    print("递归 FFT:", fr)
    print("迭代 FFT:", fi)