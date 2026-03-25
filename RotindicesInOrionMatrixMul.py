



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
    non_zero_diags = A+bsKey
    log_max_ratio =0
    bStep = _find_best_bsgs_ratio(non_zero_diags, slots, log_max_ratio)
    gStep = (slots + bStep - 1) // bStep
    # keylist=list(A.keys())
    keylist=A
    rotinpractice = set()
    for i in range(1, bStep):
        if i in keylist:
            norm_index = norm_rot_index(i,N)
            ll.append(norm_index)
            rotinpractice.add(i)
    for j in range(1, gStep):
        changed = 0
        if bStep * j in keylist:
            changed=1
            rotinpractice.add( bStep * j)
        for i in range(1,bStep):
            if  bStep*j+i in keylist:
                norm_index = norm_rot_index(i,N)
                ll.append(norm_index)
                rotinpractice.add( bStep*j+i)
                changed=1
        if changed==1:
            norm_index = norm_rot_index( bStep * j,N)
            ll.append(norm_index)
    print("rotations in practice:", sorted(rotinpractice))
    print("rotations in practice size:", len(sorted(rotinpractice)))
    print(len(keylist), len(set(keylist)))
    print(rotinpractice==set(keylist))
    print(set(keylist)-rotinpractice)
    return set(ll)


# ROT_SWK key automorphism list=[1, 5, 25, 125, 625, 1949, 2689, 3125, 3841, 4097, 4561, 7937, 8193, 9601, 9745, 10377, 11973, 12021, 12289, 13825, 13921, 15625, 16049, 16385, 16873, 16897, 17029, 17921, 18489, 18561, 18985, 20993, 21289, 21761, 22805, 24833, 24889, 25089, 25473, 25717, 25857, 28305, 28609, 28929, 29185, 29589, 31745, 32769, 33025, 33281, 34433, 37121, 37181, 37377, 38381, 38477, 39681, 39685, 41217, 41345, 41473, 42021, 44421, 45313, 45569, 45837, 46817, 47601, 48725, 49153, 49409, 49665, 50305, 51885, 52429, 53341, 53505, 54833, 57601, 59865, 60105, 60833, 61313, 61949, 63097, 63489, 65057, 65537, 66177, 66297, 67353, 67585, 68897, 69009, 69121, 69153, 69341, 69633, 69825, 71425, 71681, 73217, 73729, 74429, 75521, 75777, 77057, 77185, 77313, 77481, 77489, 77825, 78125, 79033, 79873, 81153, 81409, 81921, 82049, 82341, 82901, 82945, 83621, 84365, 84561, 85249, 86017, 86145, 87041, 89129, 89345, 90113, 90881, 91033, 91137, 92445, 93057, 93529, 93857, 94209, 94925, 94977, 95053, 95233, 96469, 97349, 97937, 98113, 98305, 99073, 99329, 102017, 102401, 102465, 103169, 106445, 106497, 106561, 106933, 107265, 108929, 110001, 110337, 110593, 112553, 113153, 114025, 114433, 114689, 116869, 117477, 117889, 118117, 118529, 118637, 118785, 121089, 122625, 122881, 124445, 124801, 125261, 126721, 126977, 128353, 128481, 128585, 130253, 130817, 131071]
bsKey=[8, 64, 512, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32320, 32384, 32448, 32512, 32576, 32640, 32704, 32712, 32720, 32728, 32736, 32744, 32752, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767]
N = 2**16
A = [0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
codesign = bsgsEvaluateLinearTransform(A, N, bsKey)
rot_list = [i for i in range(1, 67)] + [
    94, 95, 96, 97, 126, 127, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416,
    448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896,
    928, 960, 991, 992, 993, 1008, 1012, 1016, 1020, 1023, 1024, 1088, 1152,
    1216, 1792, 1856, 1920, 1984, 2048, 2112, 2176, 2240, 2816, 2880, 2944, 3008,
    3072, 3136, 3200, 3264, 3840, 3904, 3968, 4032, 4096, 4160, 4992, 5056, 5120,
    5184, 6016, 6080, 6144, 6208, 7040, 7104, 7168, 7232, 8064, 8128, 8192, 9216,
    10240, 11264, 12288, 13312, 14336, 15360, 32752, 32756, 32760, 32764,
]

print("lenof codesign:", len(set(list(codesign)+(bsKey))))
print("lenof bskey+A:", len(set(rot_list+bsKey)))

print("codesign:", sorted(set(codesign)))
print("base+bs:", set(rot_list+bsKey))

print("base=========")
base = bsgsEvaluateLinearTransform(A, N, [])
print(base)
print("base size:", len(base))
print("base+bsKey size:", len(set(list(base)+bsKey)))