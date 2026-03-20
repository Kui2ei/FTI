



def norm_rot_index(i,N):
    if i < 0:
        i = N // 2 + i
    return i

def _bsgs_index(non_zero_diags, slots, n1):
    rot_n1_map = set()
    rot_n2_map = set()

    for rot in non_zero_diags:
        rot &= (slots - 1)
        idx_n1 = ((rot // n1) * n1) & (slots - 1)
        idx_n2 = rot & (n1 - 1)
        rot_n1_map.add(idx_n1)
        rot_n2_map.add(idx_n2)
    
    # Return the number of unique rotations needed
    return len(rot_n1_map)-1, len(rot_n2_map)-1


def _find_best_bsgs_ratio(non_zero_diags, max_n, log_max_ratio):
    max_ratio = float(1 << log_max_ratio)
    n1 = 1
    while n1 < max_n:
        num_rot_n1, num_rot_n2 = _bsgs_index(non_zero_diags, max_n, n1)
        # Avoid division by zero if there's only one giant step (rotation 0)
        if num_rot_n1 == 0:
            return n1
        current_ratio = float(num_rot_n2) / float(num_rot_n1)
        if current_ratio == max_ratio:
            return n1
        if current_ratio > max_ratio:
            return n1 // 2 if n1 > 1 else 1
        n1 <<= 1    
    return  1
    
def bsgsEvaluateLinearTransform(A, N, bsKey):
    ll=[] 
    slots =  2**15
    non_zero_diags = list(A.keys())+bsKey
    log_max_ratio =0
    bStep = _find_best_bsgs_ratio(non_zero_diags, slots, log_max_ratio)
    gStep = (slots + bStep - 1) // bStep
    keylist=list(A.keys())

    for i in range(1, bStep):
        if i in keylist:
            norm_index = norm_rot_index(i,N)
            ll.append(norm_index)
    for j in range(1, gStep):
        if bStep * j in keylist:
            changed=1
        for i in range(1,bStep):
            if  bStep*j+i in keylist:
                norm_index = norm_rot_index(i,N)
                ll.append(norm_index)
                changed=1
        if changed==1:
            norm_index = norm_rot_index( bStep * j,N)
            ll.append(norm_index)
    return ll

# ROT_SWK key automorphism list=[1, 5, 25, 125, 625, 1949, 2689, 3125, 3841, 4097, 4561, 7937, 8193, 9601, 9745, 10377, 11973, 12021, 12289, 13825, 13921, 15625, 16049, 16385, 16873, 16897, 17029, 17921, 18489, 18561, 18985, 20993, 21289, 21761, 22805, 24833, 24889, 25089, 25473, 25717, 25857, 28305, 28609, 28929, 29185, 29589, 31745, 32769, 33025, 33281, 34433, 37121, 37181, 37377, 38381, 38477, 39681, 39685, 41217, 41345, 41473, 42021, 44421, 45313, 45569, 45837, 46817, 47601, 48725, 49153, 49409, 49665, 50305, 51885, 52429, 53341, 53505, 54833, 57601, 59865, 60105, 60833, 61313, 61949, 63097, 63489, 65057, 65537, 66177, 66297, 67353, 67585, 68897, 69009, 69121, 69153, 69341, 69633, 69825, 71425, 71681, 73217, 73729, 74429, 75521, 75777, 77057, 77185, 77313, 77481, 77489, 77825, 78125, 79033, 79873, 81153, 81409, 81921, 82049, 82341, 82901, 82945, 83621, 84365, 84561, 85249, 86017, 86145, 87041, 89129, 89345, 90113, 90881, 91033, 91137, 92445, 93057, 93529, 93857, 94209, 94925, 94977, 95053, 95233, 96469, 97349, 97937, 98113, 98305, 99073, 99329, 102017, 102401, 102465, 103169, 106445, 106497, 106561, 106933, 107265, 108929, 110001, 110337, 110593, 112553, 113153, 114025, 114433, 114689, 116869, 117477, 117889, 118117, 118529, 118637, 118785, 121089, 122625, 122881, 124445, 124801, 125261, 126721, 126977, 128353, 128481, 128585, 130253, 130817, 131071]
# BSrotidx_set=[8, 64, 512, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32320, 32384, 32448, 32512, 32576, 32640, 32704, 32712, 32720, 32728, 32736, 32744, 32752, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767]
N = 2**16
A = 
bsgsEvaluateLinearTransform(A, N, bsKey)
