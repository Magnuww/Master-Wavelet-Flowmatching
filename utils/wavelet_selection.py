from ast import Tuple
import math
from typing import Tuple
import numpy as np
import pywt


"ADAPTED FROM THE WAVELETDIFF PAPER"

def get_wavelet_type(seq_len : int):
    if seq_len <= 32:
        return 'db2'
    elif seq_len <= 64:
        return 'db4'
    elif seq_len <= 128:
        return 'db6'
    else:
        return 'db8'

def get_wavelet_level(seq_len : int, target_seq_len : int = 64):
    n = math.floor(math.log2(seq_len / target_seq_len))
    return int(np.clip(n, 2, 8))



if __name__ == "__main__":
    for seq_len in [24, 64, 128, 256, 512]:
        wavelet_type = get_wavelet_type(seq_len)
        wavelet_level = get_wavelet_level(seq_len, wavelet_type)
        print(f"Sequence length: {seq_len}, Selected wavelet type: {wavelet_type}, level: {wavelet_level}")
