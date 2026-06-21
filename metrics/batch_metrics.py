from metrics.context_fid import C_FID, Context_FID
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.dtw import calculate_dtw
from metrics.dcr import calculate_dcr, calculate_dcr_torch
from metrics.ed import calculate_ed
from metrics.feature_based_measures import calculate_acd, calculate_kd, calculate_mdd
from metrics.mmd import calculate_mmd
from metrics.predictive_metrics import predictive_score_metrics
from metrics.r2 import r2
import numpy as np

def compute_metrics(
    real_data,
    generated_data,
    iterations=5,
    num_samples=10000,
    metrics=[
        "discriminative_score",
        "predictive_score",
        "context_fid_score",
        "correlational_score",
        "MDD",
        "ACD",
        "KD",
        "ED",
        "MMD",
    ],
    min_max_normalize=True,
    return_full_results=False,
):
    """
    Compute evaluation metrics between real and generated time-series data.

    Assumes real_data and generated_data are NumPy arrays with shape:
        (num_sequences, sequence_length, num_features)
    """
    print(real_data.shape, generated_data.shape, num_samples)

    sample_size = min(num_samples, real_data.shape[0], generated_data.shape[0])
    print(sample_size)

    if min_max_normalize:
        print("normalizing")
        data_min = np.min(real_data, axis=(0, 1), keepdims=True)
        data_max = np.max(real_data, axis=(0, 1), keepdims=True)
        real_data = (real_data - data_min) / (data_max - data_min + 1e-8)
        generated_data = (generated_data - data_min) / (data_max - data_min + 1e-8)

    results = {}
    full_results = {}

    def sample_pair():
        real_idx = np.random.choice(real_data.shape[0], sample_size, replace=False)
        fake_idx = np.random.choice(generated_data.shape[0], sample_size, replace=False)
        return real_data[real_idx], generated_data[fake_idx]

    if "discriminative_score" in metrics:
        print("computing discriminative_score...")
        discriminative_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            temp_disc, _, _ = discriminative_score_metrics(
                real_data_sample,
                generated_data_sample,
                iterations=2000,
            )
            discriminative_scores.append(temp_disc)

        results["discriminative_score_mean"] = float(np.mean(discriminative_scores))
        results["discriminative_score_std"] = float(np.std(discriminative_scores))
        if return_full_results:
            full_results["discriminative_scores"] = discriminative_scores

    if "predictive_score" in metrics:
        print("computing predictive_score...")
        predictive_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            temp_pred = predictive_score_metrics(
                real_data_sample,
                generated_data_sample,
                iterations=5000,
            )
            predictive_scores.append(temp_pred)

        results["predictive_score_mean"] = float(np.mean(predictive_scores))
        results["predictive_score_std"] = float(np.std(predictive_scores))
        if return_full_results:
            full_results["predictive_scores"] = predictive_scores

    if "context_fid_score" in metrics:
        print("computing context_fid_score...")
        context_fid_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            context_fid = Context_FID(
                real_data_sample,
                generated_data_sample,
            )
            context_fid_scores.append(context_fid)

        results["context_fid_score_mean"] = float(np.mean(context_fid_scores))
        results["context_fid_score_std"] = float(np.std(context_fid_scores))
        if return_full_results:
            full_results["context_fid_scores"] = context_fid_scores


    if "DTW" in metrics:
        print("computing DTW...")
        dtw_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            dtw_dist = calculate_dtw(real_data_sample, generated_data_sample)
            dtw_scores.append(dtw_dist)

        results["dtw_distance_mean"] = float(np.mean(dtw_scores))
        results["dtw_distance_std"] = float(np.std(dtw_scores))
        if return_full_results:
            full_results["dtw_scores"] = dtw_scores

    if "MDD" in metrics:
        print("computing MDD...")
        mdd_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            score = calculate_mdd(real_data_sample, generated_data_sample)
            mdd_scores.append(score)

        results["mdd_score_mean"] = float(np.mean(mdd_scores))
        results["mdd_score_std"] = float(np.std(mdd_scores))
        if return_full_results:
            full_results["mdd_scores"] = mdd_scores

    if "ACD" in metrics:
        print("computing ACD...")
        acd_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            score = calculate_acd(real_data_sample, generated_data_sample)
            acd_scores.append(score)

        results["acd_score_mean"] = float(np.mean(acd_scores))
        results["acd_score_std"] = float(np.std(acd_scores))
        if return_full_results:
            full_results["acd_scores"] = acd_scores

    if "KD" in metrics:
        print("computing KD...")
        kd_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            score = calculate_kd(real_data_sample, generated_data_sample)
            kd_scores.append(score)

        results["kd_score_mean"] = float(np.mean(kd_scores))
        results["kd_score_std"] = float(np.std(kd_scores))
        if return_full_results:
            full_results["kd_scores"] = kd_scores

    if "ED" in metrics:
        print("computing ED...")
        ed_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            score = calculate_ed(real_data_sample, generated_data_sample)
            ed_scores.append(score)

        results["ed_score_mean"] = float(np.mean(ed_scores))
        results["ed_score_std"] = float(np.std(ed_scores))
        if return_full_results:
            full_results["ed_scores"] = ed_scores

    if "MMD" in metrics:
        print("computing MMD...")
        mmd_scores = []
        selfmmdscores = []
        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            real_data_sample2, _ = sample_pair()

            real_flat = real_data_sample.reshape(real_data_sample.shape[0], -1)
            real_flat2 = real_data_sample2.reshape(real_data_sample2.shape[0], -1)
            generated_flat = generated_data_sample.reshape(
                generated_data_sample.shape[0],
                -1,
            )

            selfmmd = calculate_mmd(real_flat, real_flat2)
            selfmmdscores.append(selfmmd)
            score = calculate_mmd(real_flat, generated_flat)
            mmd_scores.append(score)

        results["mmd_self_score_mean"] = float(np.mean(selfmmdscores))
        results["mmd_self_score_std"] = float(np.std(selfmmdscores))
        results["mmd_score_mean"] = float(np.mean(mmd_scores))
        results["mmd_score_std"] = float(np.std(mmd_scores))
        results["mmd_score_relative"] = float(np.mean(mmd_scores) / (np.mean(selfmmdscores) + 1e-8))
        if return_full_results:
            full_results["mmd_scores"] = mmd_scores
            full_results["mmd_self_scores"] = selfmmdscores

    if "DCR" in metrics:
        print("computing DCR...")
        score = calculate_dcr_torch(
            real_data,
            generated_data,
        )

        results["dcr_score_mean"] = float(np.mean(score))
        results["dcr_score_std"] = float(np.std(score))
        if return_full_results:
            full_results["dcr_scores"] = score.tolist()

    if "R2" in metrics:
        print("computing R2...")
        r2_scores = []

        for _ in range(iterations):
            real_data_sample, generated_data_sample = sample_pair()
            score = r2(real_data_sample, generated_data_sample)
            r2_scores.append(score)

        results["r2_score_mean"] = float(np.mean(r2_scores))
        results["r2_score_std"] = float(np.std(r2_scores))
        if return_full_results:
            full_results["r2_scores"] = r2_scores

    if return_full_results:
        return full_results
    return results



def compute_metrics_recon(real_data, generated_data, metrics = ["mae", "mape", "rel", "r2", "ed", "mmd", "acd", "kd", "mdd", "psnr", "snr"]):
    results = {}
    for m in metrics:
        if m in ("mape", "percentage"):
            denom = np.abs(real_data)
            pct = np.abs(generated_data - real_data) / (denom + 1e-8)
            out = np.mean(pct)
            results['mape'] = float(out * 100.0)

        if m == "rel":
            num = np.linalg.norm(generated_data - real_data)
            den = np.linalg.norm(real_data)
            results['rel'] = float(num / (den + 1e-8))

        if m == "r2":
            results['r2'] = float(r2(real_data, generated_data))

        if m == "ed":
            results['ed'] = float(calculate_ed(real_data, generated_data))

        if m == "acd":
            results['acd'] = float(calculate_acd(real_data, generated_data))

        if m == "kd":
            results['kd'] = float(calculate_kd(real_data, generated_data))

        if m == "mdd":
            results['mdd'] = float(calculate_mdd(real_data, generated_data))

        if m == "psnr":
            mse = np.mean((generated_data - real_data) ** 2)
            max_val = np.max(real_data)
            psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-8))
            results['psnr'] = float(psnr)

        if m == "snr":
            signal_power = np.mean(real_data ** 2)
            noise_power = np.mean((generated_data - real_data) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
            results['snr'] = float(snr)

    return results
