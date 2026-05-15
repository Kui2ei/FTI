# 来源于OPENFHE，最原始的rotate index选取逻辑'
# BSGS
import math


    # std::vector<uint32_t> levelBudget1 = {4, 4};
    # std::vector<uint32_t> levelBudget2 = {2, 4};
    # std::vector<uint32_t> levelBudget3 = {3, 2};
    # std::vector<uint32_t> levelBudget4 = {1, 1};
    # std::vector<uint32_t> levelBudget5 = {1, 2};
    # std::vector<uint32_t> levelBudget6 = {3, 1};


def selectLayers(logslots, levelBudget):
    # 每一个level里的FFT（折叠：折叠layers层FFT，消耗一层level）
    #levelbudget是一共可以消耗的level，将不同的FFT层数折叠起来塞进一层level中，共有logslot层FFT，layer代表每消耗一层level需要对应多少层FFT
    layers = math.ceil(logslots/levelBudget) 
    # 有rows个level，每个level里有layers层FFT，有rows个折叠
    rows = int(logslots // layers)
    # 最后一个rows里面不足layers层FFT，而是rem层FFT
    rem = logslots % layers
    # 真正的折叠后层数（collapsed layers）
    dim = rows + (1 if rem > 0 else 0)
    
    # 没用完levelbudget，可以多折叠一层FFT
    if(dim<levelBudget):
        layers-=1
        rows = int(logslots // layers)
        rem = logslots  - rows * layers
        dim = rows + (1 if rem > 0 else 0)
        if(dim>levelBudget):
            while dim!=levelBudget:
                rows-=1
                rem = logslots  - rows * layers
                dim = rows + (1 if rem > 0 else 0)
    print(f"layers: {layers}, rows: {rows}, rem: {rem}")
    return {'layers':layers,'rows':rows,'rem':rem}

def GetCollapsedFFTParams(slots,levelBudget = 4, dim1 = 0):
    if slots == 0:
        raise ValueError("slots must be greater than 0")
    if levelBudget ==0:
        raise ValueError("levelBudget must be greater than 0")
    logslots = 1 if slots<3 else int(math.log2(slots))
    dims = selectLayers(logslots, levelBudget)
    layersCollapse = dims['layers']
    remCollapse = dims['rem']

    # 每消耗一个level需要numRotation个rotation，最后一层需要numRotationRem个rotation
    numRotations = (1 << (layersCollapse + 1)) - 1
    numRotationsRem = (1 << (remCollapse + 1)) - 1

    shiftBase = layersCollapse // 2 + 1 + (1 if numRotations > 7 else 0)
    print("shiftBase: ", shiftBase)
    gDefault = 1 << shiftBase
    g = gDefault if (dim1 == 0 or dim1 > numRotations) else dim1
    # g = math.ceil(math.pow(numRotations,0.5))
    b = (numRotations+1) // g

    if remCollapse != 0:
        shiftBaseRem = remCollapse // 2 + 1 + (1 if numRotationsRem > 7 else 0)
        gRem = 1 << shiftBaseRem
        bRem = (numRotationsRem + 1) // gRem
    else:
        gRem = 0
        bRem = 0
    
    return {"levelBudget":levelBudget,
            "layersCollapse":layersCollapse,
            "remCollapse":remCollapse,
            "numRotations":numRotations,
            "b":b,
            "g":g,
            "numRotationsRem":numRotationsRem,
            "bRem":bRem,
            "gRem":gRem}
            
        
def IsPowerOfTwo(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def ReduceRotation(index: int, slots: int) -> int:
    islots = int(slots)
    if islots <= 0:
        raise ValueError("slots must be positive")
    if IsPowerOfTwo(islots):
        n = islots.bit_length() - 1
        if index >= 0:
            return index - ((index >> n) << n)
        return index + islots + ((abs(index) >> n) << n)

    return (islots + index % islots) % islots
        
def FindCoeffsToSlotsRotationIndices(slots: int, M: int, lb:int, dim1:int = 0):
    slots = int(slots)
    M = int(M)
    params = GetCollapsedFFTParams(slots,lb,dim1)

    levelBudget = int(params["levelBudget"])
    layersCollapse = int(params["layersCollapse"])
    remCollapse = int(params["remCollapse"])
    numRotations = int(params["numRotations"])
    b = int(params["b"])
    g = int(params["g"])
    numRotationsRem = int(params["numRotationsRem"])
    bRem = int(params["bRem"])
    gRem = int(params["gRem"])

    flagRem = 0 if remCollapse == 0 else 1
    
    indexList = []
    indexListSz = b + g - 2 + bRem + gRem - 2 + 1 + M
    if indexListSz < 0:
        raise ValueError("indexListSz can not be negative")

    for s in range(levelBudget - 1, flagRem - 1, -1):
        scalingFactor = 1 << ((s - flagRem) * layersCollapse + remCollapse)
        halfRots = 1 - ((numRotations + 1) // 2)
        for j in range(halfRots, g + halfRots):
            indexList.append(ReduceRotation(j * scalingFactor, slots))
        for i in range(b):
            indexList.append(ReduceRotation((g * i) * scalingFactor, M // 4))

    if flagRem:
        halfRots = 1 - ((numRotationsRem + 1) // 2)
        for j in range(halfRots, gRem + halfRots):
            indexList.append(ReduceRotation(j, slots))
        for i in range(bRem):
            indexList.append(ReduceRotation(gRem * i, M // 4))

    m = slots * 4
    if m != M:
        ratio = M // m
        j = 1
        while j < ratio:
            indexList.append(j * slots)
            j <<= 1

    return indexList


def FindSlotsToCoeffsRotationIndices(slots: int, M: int, lb:int, dim1:int = 0):
    slots = int(slots)
    M = int(M)
    params = GetCollapsedFFTParams(slots,lb,dim1)

    levelBudget = int(params["levelBudget"])
    layersCollapse = int(params["layersCollapse"])
    remCollapse = int(params["remCollapse"])
    numRotations = int(params["numRotations"])
    b = int(params["b"])
    g = int(params["g"])
    numRotationsRem = int(params["numRotationsRem"])
    bRem = int(params["bRem"])
    gRem = int(params["gRem"])

    flagRem = 0 if remCollapse == 0 else 1
    if levelBudget < flagRem:
        raise ValueError("levelBudget can not be less than flagRem")

    indexList = []
    indexListSz = b + g - 2 + bRem + gRem - 2 + 1 + M
    if indexListSz < 0:
        raise ValueError("indexListSz can not be negative")

    for s in range(0, levelBudget - flagRem):
        scalingFactor = 1 << (s * layersCollapse)
        halfRots = 1 - ((numRotations + 1) // 2)
        for j in range(halfRots, g + halfRots):
            indexList.append(ReduceRotation(j * scalingFactor, M // 4))
        for i in range(b):
            indexList.append(ReduceRotation((g * i) * scalingFactor, M // 4))

    if flagRem:
        s = levelBudget - flagRem
        scalingFactor = 1 << (s * layersCollapse)
        halfRots = 1 - ((numRotationsRem + 1) // 2)
        for j in range(halfRots, gRem + halfRots):
            indexList.append(ReduceRotation(j * scalingFactor, M // 4))
        for i in range(bRem):
            indexList.append(ReduceRotation((gRem * i) * scalingFactor, M // 4))

    m = slots * 4
    if m != M:
        ratio = M // m
        j = 1
        while j < ratio:
            indexList.append(j * slots)
            j <<= 1
    return indexList

def FindLinearTransformRotationIndices(slots: int, M: int, dim1:int = 0):
    slots = int(slots)
    M = int(M)
    g = int(math.ceil(math.sqrt(slots))) if dim1 == 0 else int(dim1)
    h = int(math.ceil(slots / g))

    indexList = []
    for i in range(1, g + 1):
        indexList.append(i)
    for i in range(2, h):
        indexList.append(g * i)

    m = slots * 4
    if m != M:
        j = 1
        while j < M // m:
            indexList.append(j * slots)
            j <<= 1

    return indexList

def FindBootstrapRotationIndices(slots: int, M: int, lb1:int, lb2:int, dim1 = [0,0]):
    res = []
    if lb1 == 1 and lb2 == 1:
        res.extend(FindLinearTransformRotationIndices(slots, M, dim1[0]))
    else:
        res.extend(FindCoeffsToSlotsRotationIndices(slots, M, lb1, dim1[0]))
        print(f"FindCoeffsToSlotsRotationIndices: ", sorted(set(res)))
        a = FindSlotsToCoeffsRotationIndices(slots, M, lb2, dim1[1])
        print(f"FindSlotsToCoeffsRotationIndices: ",sorted(set(a)))
        res.extend(a)
    res = set(res)
    res.discard(0)
    res.discard(M//4)
    return list(sorted(res))

def RotInOPENFHE(logN,logSlots,levelBudget = [3,3], dim1 = [0,0]):
    if levelBudget[0] > logSlots or levelBudget[1] > logSlots:
        raise ValueError("levelBudget must be at most logSlots")
    if levelBudget[0] < 1 or levelBudget[1] < 1:
        raise ValueError("levelBudget must be at least 1")
    slots = 1 << logSlots
    N = 1 << logN
    return FindBootstrapRotationIndices(slots, N*2, levelBudget[0], levelBudget[1], dim1)


a = RotInOPENFHE(7,6)  # Example usage
print(a)
print(len(a))
