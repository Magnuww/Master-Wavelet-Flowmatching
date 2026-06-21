import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
FROM TSG_BENCH
adapt from https://github.com/jsyoon0823/TimeGAN, https://openreview.net/forum?id=ez6VHWvuXEx
"""

def visualize_tsne(ori_data, gen_data, mean=True, perplexity=30):
    sample_num = min([1000, len(ori_data), len(gen_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]
    idx2 = np.random.permutation(len(gen_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx2]

    if mean:
        prep_data = np.mean(ori_data, axis=1)
        prep_data_hat = np.mean(gen_data, axis=1)
    else:
        prep_data = ori_data.reshape(sample_num, -1)
        prep_data_hat = gen_data.reshape(sample_num, -1)

    colors = ["C0" for i in range(sample_num)] + ["C1" for i in range(sample_num)]    
    
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    tsne = TSNE(n_components = 2, verbose = 0, perplexity = perplexity, max_iter = 1000, random_state = 42) # 40, 300
    tsne_results = tsne.fit_transform(prep_data_final)

    fig, ax = plt.subplots(1,1)
    
    ax.scatter(tsne_results[:sample_num,0], tsne_results[:sample_num,1], 
                c = colors[:sample_num], alpha = 0.5, label = "Original", s = 5)
    ax.scatter(tsne_results[sample_num:,0], tsne_results[sample_num:,1], 
                c = colors[sample_num:], alpha = 0.5, label = "Generated", s = 5)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)
    plt.close(fig)
    return fig
