from numba import cfunc, types, carray, njit

numba_signature = types.float32(
    types.CPointer(types.uint32), types.CPointer(types.uint32)
)


@njit("int_(uint32)")
def word_popcount(v):
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    c = types.uint32((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24
    return c


@cfunc(numba_signature)
def tanimoto_maccs(a, b):
    a_array = carray(a, 6)
    b_array = carray(b, 6)
    ands = 0
    ors = 0
    ands += word_popcount(a_array[0] & b_array[0])
    ors += word_popcount(a_array[0] | b_array[0])
    ands += word_popcount(a_array[1] & b_array[1])
    ors += word_popcount(a_array[1] | b_array[1])
    ands += word_popcount(a_array[2] & b_array[2])
    ors += word_popcount(a_array[2] | b_array[2])
    ands += word_popcount(a_array[3] & b_array[3])
    ors += word_popcount(a_array[3] | b_array[3])
    ands += word_popcount(a_array[4] & b_array[4])
    ors += word_popcount(a_array[4] | b_array[4])
    ands += word_popcount(a_array[5] & b_array[5])
    ors += word_popcount(a_array[5] | b_array[5])
    return 1 - types.float32(ands) / ors


@cfunc(numba_signature)
def tanimoto_ecfp4(a, b):
    a_array = carray(a, 64)
    b_array = carray(b, 64)
    ands = 0
    ors = 0
    for i in range(64):
        ands += word_popcount(a_array[i] & b_array[i])
        ors += word_popcount(a_array[i] | b_array[i])
    return 1 - types.float32(ands) / ors


@cfunc(numba_signature)
def tanimoto_mixed(a, b):
    a_array = carray(a, 6 + 64)
    b_array = carray(b, 6 + 64)

    ands = 0
    ors = 0
    for i in range(6 + 64):
        ands += word_popcount(a_array[i] & b_array[i])
        ors += word_popcount(a_array[i] | b_array[i])
    return 1 - types.float32(ands) / ors


@cfunc(numba_signature)
def tanimoto_conditional(a, b):
    threashold = 0.2
    a_array = carray(a, 6 + 64)
    b_array = carray(b, 6 + 64)

    ands_maccs = 0
    ors_maccs = 0
    for i in range(6):
        ands_maccs += word_popcount(a_array[i] & b_array[i])
        ors_maccs += word_popcount(a_array[i] | b_array[i])
    maccs = 1 - types.float32(ands_maccs) / ors_maccs
    if maccs > threashold:
        return maccs

    ands_ecfp4 = 0
    ors_ecfp4 = 0
    for i in range(64):
        ands_ecfp4 += word_popcount(a_array[6 + i] & b_array[6 + i])
        ors_ecfp4 += word_popcount(a_array[6 + i] | b_array[6 + i])
    ecfp4 = 1 - types.float32(ands_ecfp4) / ors_ecfp4

    return ecfp4 * threashold
