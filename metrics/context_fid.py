import scipy
import numpy as np

from metrics.ts2vec_model.ts2vec import TS2Vec
from metrics.ts2vec import initialize_ts2vec
import torch
import time


"""

Code for calculating Context_FID

Based on the implementation from the papers:

Sigdiffusions: https://arxiv.org/abs/2406.10354

Waveletdiff: https://arxiv.org/abs/2510.11839

    """

def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def Context_FID(ori_data, generated_data):
    start = time.time()
    model = TS2Vec(input_dims=ori_data.shape[-1], device="cuda", batch_size=8, lr=0.001, output_dims=320,
                   max_train_length=3000)
    print("fitting TS2Vec model...")
    model.fit(ori_data, verbose=False)
    print("done fitting, encoding data...")
    ori_represenation = model.encode(ori_data, encoding_window='full_series')
    gen_represenation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    end = time.time()
    print(f"Context FID: {results:.4f}, time taken: {end - start:.2f} seconds")
    return results


def C_FID(ori_data, gen_data):
    fid_model = initialize_ts2vec(ori_data,"cuda")
    ori_repr = fid_model.encode(ori_data, encoding_window='full_series')
    gen_repr = fid_model.encode(gen_data, encoding_window='full_series')
    score =  calculate_fid(ori_repr,gen_repr)
    print(f"Context FID: {score:.4f}")
    return score

