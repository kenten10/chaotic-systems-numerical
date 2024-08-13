# Utility functions

# xが2の冪乗か
def is_pow2(x):
    return (x != 0) and (x & (x - 1) == 0)