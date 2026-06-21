"""

FROM TSGBENCH

    """
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
import numpy as np
def calculate_dtw(ori_data,comp_data):
    distance_dtw = []
    n_samples = ori_data.shape[0]
    for i in range(n_samples):
        distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
        distance_dtw.append(distance)

    distance_dtw = np.array(distance_dtw)
    average_distance_dtw = distance_dtw.mean()
    return average_distance_dtw
