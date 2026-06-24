import numpy as np
"numpy MMD IMPLEMENTATION ADAPTED FROM https://www.onurtunali.com/ml/2020/03/08/maximum-mean-discrepancy-in-machine-learning.html"
from sklearn.metrics.pairwise import rbf_kernel


def calculate_mmd(X, Y, squared=False):
    gammas = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]

    XX = np.zeros((X.shape[0], X.shape[0]))
    YY = np.zeros((Y.shape[0], Y.shape[0]))
    XY = np.zeros((X.shape[0], Y.shape[0]))

    for gamma in gammas:
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
        XX += K_XX
        YY += K_YY
        XY += K_XY

    XX /= len(gammas)
    YY /= len(gammas)
    XY /= len(gammas)

    mmd = np.mean(XX + YY - 2 * XY)
    if squared:
        return mmd
    else:
        return np.sqrt(max(mmd,0))
