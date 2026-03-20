# 来源于HEAANBOOT，最原始的rotate index选取逻辑'
# BSGS
import math

def RotInHEAANBOOT(logN,logSlots):
    N = 1 << logN
    Slots = 1 << logSlots
    logk = logSlots // 2
    k = 1 << logk
    m = 1<<(logSlots-logk)
    logNh = logN -1
    
    RotIndexList = []
    # part1, 2^0, 2^1, ..., 2^(logNh-1)
    for i in range(logNh):
        RotIndexList.append(1 << i)
        
    # part2 1 to k-1
    for i in range(1,k):
        RotIndexList.append(i)

    # part3 k, 2k, ..., (m-1)k
    for i in range(1,m):
        idx = i*k
        RotIndexList.append(idx)
        
    return sorted(set(RotIndexList))

print(RotInHEAANBOOT(15,4))