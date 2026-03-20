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
    numRotationsRem = (1 << remCollapse) - 1

    shiftBase = remCollapse // 2 + 1 + (1 if numRotationsRem > 7 else 0)
    gDefault = 1 << shiftBase
    g = gDefault if (dim1 == 0 or dim1 > numRotations) else dim1
    b = numRotations // g

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
        
def FindCoeffsToSlotsRotationIndices(slots: int, M: int, lb:int):
    slots = int(slots)
    M = int(M)
    params = GetCollapsedFFTParams(slots,lb)

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


def FindSlotsToCoeffsRotationIndices(slots: int, M: int, lb:int):
    slots = int(slots)
    M = int(M)
    params = GetCollapsedFFTParams(slots,lb)

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

def FindBootstrapRotationIndices(slots: int, M: int, lb:int):
    res = []
    res.extend(FindCoeffsToSlotsRotationIndices(slots, M, lb))
    res.extend(FindSlotsToCoeffsRotationIndices(slots, M, lb))
    res = set(res)
    res.discard(0)
    res.discard(M//4)
    return list(sorted(res))
        
        
#             void EvalBootstrapKeyGen(const PrivateKey<Element> privateKey, uint32_t slots) {
#         ValidateKey(privateKey);
#         auto evalKeys = GetScheme()->EvalBootstrapKeyGen(privateKey, slots);
#         CryptoContextImpl<Element>::InsertEvalAutomorphismKey(evalKeys, privateKey->GetKeyTag());
#     }
            
            
        
#         std::shared_ptr<std::map<uint32_t, EvalKey<DCRTPoly>>> FHECKKSRNS::EvalBootstrapKeyGen(
#     const PrivateKey<DCRTPoly> privateKey, uint32_t slots) {
#     const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(privateKey->GetCryptoParameters());

#     if (cryptoParams->GetKeySwitchTechnique() != HYBRID)
#         OPENFHE_THROW("CKKS Bootstrapping is only supported for the Hybrid key switching method.");
# #if NATIVEINT == 128
#     if (cryptoParams->GetScalingTechnique() == FLEXIBLEAUTO || cryptoParams->GetScalingTechnique() == FLEXIBLEAUTOEXT)
#         OPENFHE_THROW("128-bit CKKS Bootstrapping is supported for FIXEDMANUAL and FIXEDAUTO methods only.");
# #endif

#     auto cc   = privateKey->GetCryptoContext();
#     auto algo = cc->GetScheme();
#     auto M    = cc->GetCyclotomicOrder();

#     if (slots == 0)
#         slots = M / 4;

#     // computing all indices for baby-step giant-step procedure
#     auto evalKeys = algo->EvalAtIndexKeyGen(nullptr, privateKey, FindBootstrapRotationIndices(slots, M));

#     auto conjKey       = ConjugateKeyGen(privateKey);
#     (*evalKeys)[M - 1] = conjKey;

#     if (cryptoParams->GetSecretKeyDist() == SPARSE_ENCAPSULATED) {
#         DCRTPoly::TugType tug;
#         DCRTPoly sNew(tug, cryptoParams->GetElementParams(), Format::EVALUATION, 32);

#         // sparse key used for the modraising step
#         auto skNew = std::make_shared<PrivateKeyImpl<DCRTPoly>>(cc);
#         skNew->SetPrivateElement(std::move(sNew));

#         // we reserve M-4 and M-2 for the sparse encapsulation switching keys
#         // Even autorphism indices are not possible, so there will not be any conflict
#         (*evalKeys)[M - 4] = KeySwitchGenSparse(privateKey, skNew);
#         (*evalKeys)[M - 2] = algo->KeySwitchGen(skNew, privateKey);
#     }

#     return evalKeys;
# }
        
        
        
        
        
#         std::vector<int32_t> FHECKKSRNS::FindBootstrapRotationIndices(uint32_t slots, uint32_t M) {
#     auto& p = GetBootPrecom(slots);
#     bool isLTBootstrap =
#         (p.m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) && (p.m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1);

#     std::vector<uint32_t> fullIndexList;
#     if (isLTBootstrap) {
#         fullIndexList = FindLinearTransformRotationIndices(slots, M);
#     }
#     else {
#         fullIndexList = FindCoeffsToSlotsRotationIndices(slots, M);

#         std::vector<uint32_t> indexListStC{FindSlotsToCoeffsRotationIndices(slots, M)};
#         fullIndexList.insert(fullIndexList.end(), std::make_move_iterator(indexListStC.begin()),
#                              std::make_move_iterator(indexListStC.end()));
#     }

#     // Remove possible duplicates and remove automorphisms corresponding to 0 and M/4 by using std::set
#     std::set<uint32_t> s(fullIndexList.begin(), fullIndexList.end());
#     s.erase(0);
#     s.erase(M / 4);

#     return std::vector<int32_t>(s.begin(), s.end());
# }
        
        
        
        
        
        
        
#         void CryptoContextImpl<Element>::InsertEvalAutomorphismKey(
#     const std::shared_ptr<std::map<uint32_t, EvalKey<Element>>> mapToInsert, const std::string& keyTag) {
#     // check if the map is empty
#     if (mapToInsert->empty()) {
#         return;
#     }

#     auto mapToInsertIt    = mapToInsert->begin();
#     const std::string& id = (keyTag.empty()) ? mapToInsertIt->second->GetKeyTag() : keyTag;
#     std::set<uint32_t> existingIndices{CryptoContextImpl<Element>::GetExistingEvalAutomorphismKeyIndices(id)};
#     if (existingIndices.empty()) {
#         // there is no keys for the given id, so we insert full mapToInsert
#         CryptoContextImpl<Element>::s_evalAutomorphismKeyMap[id] = mapToInsert;
#     }
#     else {
#         // get all indices from mapToInsert
#         std::set<uint32_t> newIndices;
#         for (const auto& [key, _] : *mapToInsert) {
#             newIndices.insert(key);
#         }

#         // find all indices in mapToInsert that are not in the exising map and
#         // insert those new indices and their corresponding keys to the existing map
#         std::set<uint32_t> indicesToInsert{CryptoContextImpl<Element>::GetUniqueValues(existingIndices, newIndices)};
#         auto keyMapIt = CryptoContextImpl<Element>::s_evalAutomorphismKeyMap.find(id);
#         auto& keyMap  = *(keyMapIt->second);
#         for (uint32_t indx : indicesToInsert) {
#             keyMap[indx] = (*mapToInsert)[indx];
#         }
#     }
# }
        
        
        
        
        
def RotInOPENFHE(logN,logSlots,levelBudget = [5,5], dim1 = [0,0]):
    slots = 1 << logSlots
    N = 1 << logN
    return FindBootstrapRotationIndices(slots, N*2, levelBudget[0])


a = RotInOPENFHE(16,15)  # Example usage
print(a)
print(len(a))