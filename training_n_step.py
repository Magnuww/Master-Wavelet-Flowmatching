import re
from unicodedata import normalize

from matplotlib.animation import FuncAnimation
from pandas.compat.pyarrow import pa
from sklearn.preprocessing import StandardScaler
from data.data_loader import RICODataset, RICODatasetDict, SineDataset, SineDatasetDict, get_dataset, FILEDatasetDict, FILEDataset
from data.data_configs import get_data_config
from fm_training import evaluate_model_core, sample_nstep_flow_freq_domain, train_n_step_flow_matching_freq_domain
from metrics.visualize_tsne import visualize_tsne
from models.lossfuncs import MSE, get_loss_func   
from models.models import ChannelMLPODE, CondMLPODE4, IdentityODE, MLPODE3_multichannel, MLPODE4, MixerODE, ShapedPriorModel, SimpleMLPODE, TransformerODE, UnetODE
from models.transforms import FlatWaveletTransformer, IdentityTransformer, Transform, automaticMaskTransform
from utils.stats import compute_channel_stats
from models.transform_factory import get_transform
from pandas.core.groupby.generic import _transform_template
from pywt import data
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from utils.logger import EmptyLogger, Logger, MlFlowLogger, WandbLogger
from utils.plot_utils import format_coeffs_for_plotting, plot_distribution, plot_multiple_multivariate_samples_togheter_subfigure, plot_multiple_samples_togheter_subfigure, plot_n_batches_vertical 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from utils.wavelet_selection import get_wavelet_level, get_wavelet_type

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    if device == "cuda":
        try:
            idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_properties(idx).name
        except Exception:
            gpu_name = None
    print("GPU:", gpu_name if gpu_name else "none detected")
    parser = argparse.ArgumentParser()
    split_group = parser.add_mutually_exclusive_group()
    parser.add_argument('-c', '--cpu', action='store_true',
                        help='Force use of CPU')
    parser.add_argument('-o', '--output', type=str, metavar='PATH',
                        help='Output path for figure')
    parser.add_argument('-m', '--mlflow_path' ,type=str )
    parser.add_argument('-d', '--mlflow_db' ,type=str )
    parser.add_argument(
        '--logger',
        type=str,
        choices=['mlflow', 'wandb'],
        help='Specify which logger to use: "mlflow" or "wandb".'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify which logger to use: "mlflow" or "wandb".'
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
    help="Window size for the dataset (default: 64)"
    )
    parser.add_argument(
        "--lr"
        , type=float,
        default=1e-4,
    help="Learning rate for training (default: 1e-4)"
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Alpha value for combined loss (default: 0.5)'
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=-1,
        help="Number of wavelet decomposition levels (default: -1 aka auto)"
    )
    parser.add_argument(
        "--wavelet_type",
        type=str,
        default="auto",
        help="Type of wavelet to use for wavelet transform (default: db4)"
    )
    parser.add_argument(
        "--disable_eval",
        action="store_true",
        help="Whether to disable evaluation after training (default: False)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=2000,
    help="Number of training epochs (default: 2000)"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional group name for organizing runs in the logger (default: None)"
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        choices=[None, "minmax", "z"],
        help="Method to normalize input data for better training stability (default: None, which means no normalization). Options are 'minmax', 'z'." 
    )
    parser.add_argument(
        "--cond",
        action="store_true",
        help="Should the second model be conditioned"
    )
    parser.add_argument(
        "--disable_ot",
        action="store_true",
        help="Whether to disable optimal transport pairing during training"
    )

    parser.add_argument(
        "--coeff_norm",
        type=str,
        default=None,
        choices=[None, "minmax", "z"],
        help="Method to normalize wavelet coefficients for better training stability (default: None, which means no normalization). Options are'minmax', 'z'" 
    )
    parser.add_argument(
            "--global_coeff_norm",
            action="store_true",
            help="If true, apply the same normalization to all coefficients based on the global min and max or mean and std of all coefficients. If false, apply normalization separately for each coefficient group based on their own min and max or mean and std. This option is ignored if --multi_coeff_norm is provided, as multi coeff norm assumes separate normalization for each group."
    )
    parser.add_argument(
        "--multi_coeff_norm",
        type=int,
        nargs="+",
        default=None,
        help="list of splits to apply different normalization to different coefficient groups")
        
    split_group.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=None,
        help="cumulative list of splits to divide the wavelt, if not provided uses the wavelets natural splits"
    )
    parser.add_argument(
        "--keep_n_groups",
        type=int,
        default=None,
        help="Keep the n first groups of coefficients together in one model, can only be run wiht --no_split"
    )
    split_group.add_argument(
        "--no_split",
        action="store_true",
        help="training with only one model, equivalent to passing one split with the total number of coefficients, can be used together with --splits to add a final split if the provided splits do not sum up to the total number of coefficients"
        )

    split_group.add_argument(
        "--half_split"
        , action="store_true",
        help="If true, split the coefficients in two halves, with the first half containing the low frequency coefficients and the second half containing the high frequency coefficients. This is a common split used in wavelet-based models, as it allows the model to learn separate dynamics for low and high frequency components."
    )
    split_group.add_argument(
        "--level_split",
        type=int,
        nargs="+",
        default=None,
        help="Split the coefficients based on wavelet decomposition levels. Provide a list of level indexes where to split. For example, if levels=3 and level_split=[1], the first model will be trained on the coefficients from level 3 (the lowest frequency coefficients) and the second model will be trained on the coefficients from levels 1 and 2 (the higher frequency coefficients). If level_split=[2], the first model will be trained on the coefficients from levels 2 and 3, and the second model will be trained on the coefficients from level 1."
    )

    split_group.add_argument(
        "--tripple_split",
        action="store_true",
        help="Split the coefficients in three groups, with the first group containing the low frequency coefficients, the second group containing the mid frequency coefficients and the third group containing the high frequency coefficients. The split is done based on the natural splits of the wavelet coefficients, so the first group will contain the approximation coefficients"
    )

    parser.add_argument(
        "--drop_remainders",
        action="store_true",
        help="If true, drop any remaining coefficients that are not included in the provided splits."
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "new_recon_loss", "new_combined_loss"],
        )

    parser.add_argument(
        "--artifact_folder",
        type=str,
        default="runs",
        help="Base directory for storing logs and results (default: 'runs')"
    )
    parser.add_argument(
        "--model_class",
        type=str,
        nargs="+",
        default="SimpleMLPODE",
        choices=["SimpleMLPODE",  "UnetODE", "TransformerODE", "ChannelMLPODE", "MixerODE", "IdentityODE"],
        )
    parser.add_argument(
        "--transform",
        type=str,
        default="flatwavelet",
        choices=["flatwavelet", "identity"],
        help="Type of transform to apply to the data before feeding it to the model (default: flatwavelet). Options are 'flatwavelet' and 'identity'."
    )

    parser.add_argument(
        "--shaped_prior",
        action="store_true",
        help="wheter to use shaped prior for sampling and training",
    )

    parser.add_argument(
        "--target_seq_len",
        type=int,
        default=64,
        help="Target sequence length for wavelet decomposition, used to automatically determine the number of levels and wavelet type if not provided. Default is 64, which means the wavelet will be decomposed until the coefficients have a length of 64 or less."
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else (
        "cuda" if torch.cuda.is_available() else "cpu")

    window_size = args.window_size
    dataset = args.dataset
    wavelet_type = args.wavelet_type if args.wavelet_type != "auto" else get_wavelet_type(window_size)
    levels = args.levels if args.levels != -1 else get_wavelet_level(window_size, args.target_seq_len)
    coeff_normalize = args.coeff_norm
    coeff_multi_norm = args.multi_coeff_norm
    coeff_norm_requested = coeff_normalize is not None
    coeff_bool = False
    splits = args.splits
    no_split = args.no_split
    assert not (coeff_multi_norm is not None and no_split is False), "Cannot use multi_coeff_norm when no_split is false, as multi coeff level assumes just one model"
    assert not (args.global_coeff_norm and coeff_normalize is None), (
        "Global coefficient normalization requires a normalization method to be specified with --coeff_norm, but got None. "
        "Please specify a normalization method (e.g., --coeff_norm minmax) or disable global coefficient normalization with --no_global_coeff_norm."
    )
    assert not (args.keep_n_groups is not None and no_split is False), "can only use keep_n_groups together with no_split, as keep_n_groups is only defined for one model"

    group = args.group
    cond = args.cond
    artifact_folder = args.artifact_folder
    norm = args.norm.upper() if args.norm is not None else None

    datasetConf = get_data_config(dataset, window_size=window_size,normalize=norm, transformer=None)

    eval = None if args.disable_eval else {
        "method_list" : ["discriminative_score", "predictive_score", "context_fid_score", "correlational_score", "MDD", "ACD", "KD", "ED", "MMD", "DTW"],
    }


    config = {
        "device": device,
        "log_model" : False,
        'epochs': args.epochs,
        'batch_size': 128,
        'lr': args.lr,
        'weight_decay': 1e-4,
        "log_per_epoch" : None,
        "eval_per_epoch" : 5000,
        "dataset" : datasetConf,
        "disable_ot" : args.disable_ot,
        "cond_noise_dropout" : {
            "std" : 0.15,
            "dropout" : 0.1,
        },
        "eval" : eval,
            "during_training_eval_metrics" : {"method_list" : ["correlational_score", "MDD", "ACD", "KD", "ED", "MMD", "DTW"]}
    }

    dataset = get_dataset(datasetConf)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    _, sample = next(iter(dataloader))
    # transformer = get_transform(dataloader, transform, input_example=sample, wavelet_type=args.wavelet_type, levels=levels, topk=topk_mask)
    # lossfunc = get_loss_func(loss, transform=transformer, alpha=alpha)

    base_transformer = get_transform(dataloader, transform_name=args.transform, input_example=sample, wavelet_type=wavelet_type, levels=levels)
    tr_sample = base_transformer.transform(sample)

    if isinstance(base_transformer, FlatWaveletTransformer):
        print(f"Using FlatWaveletTransformer with wavelet type {wavelet_type} and levels {levels}")
        shapes = base_transformer.shapes
    elif isinstance(base_transformer, IdentityTransformer):
        shapes = [tr_sample.shape[2]]
    else :
        raise ValueError(f"Unsupported transformer type: {type(base_transformer)}")

    total_coeffs = sum(shapes)
    print(f"Base transformer shapes: {shapes}, total = {total_coeffs}")
    if no_split:
        splits = [tr_sample.shape[2]]

    if args.half_split:
        cumsum = np.cumsum(shapes)
        splits = [cumsum[-2]]
        print(splits)

    if args.level_split is not None:
        cumsum = np.cumsum(shapes)
        assert max(args.level_split) <= levels and min(args.level_split) >= 1, f"Level split indexes must be between 1 and {levels}. Got {args.level_split}"
        assert len(args.level_split) <= levels, f"Number of level splits must be less than the number of levels. Got {len(args.level_split)} splits for {levels} levels."
        splits = [cumsum[level - 1] for level in args.level_split]
        print(f"Using level-based splits at levels {args.level_split}: {splits}")

    if coeff_multi_norm and coeff_multi_norm[0] == -1 and len(coeff_multi_norm) == 1:
        coeff_multi_norm = shapes

    if splits is not None:
        splits = list(splits)
        if any(split < 0 for split in splits):
            raise ValueError(f"Split boundaries must be non-negative, got {splits}")
        if any(left >= right for left, right in zip(splits, splits[1:])):
            raise ValueError(f"Split boundaries must be strictly increasing, got {splits}")
        if splits[-1] == 0:
            raise ValueError(f"At least one coefficient must be included, got {splits}")
        if splits[-1] > total_coeffs:
            raise ValueError(f"Last split boundary {splits[-1]} exceeds total coefficients {total_coeffs}")
        if splits[-1] < total_coeffs:
            if args.drop_remainders:
                print(f"Dropping coefficient remainder {splits[-1]}:{total_coeffs}")
            else:
                print(f"Warning: Provided splits {splits} do not sum up to the total number of coefficients {total_coeffs}. adding the last split to match the total.")
                splits.append(total_coeffs)
        print(f"Using custom splits: {splits}")
        if splits[0] != 0:
            indx = np.concatenate(([0], splits))
        else:
            indx = np.array(splits)
    else:
        splits = shapes
        cumsum = np.cumsum(shapes)
        indx  = np.concatenate(([0], cumsum))

    if args.tripple_split:
        assert len(shapes) >= 3, f"Tripple split requires at least 3 groups of coefficients, but got {len(shapes)} groups with shapes {shapes}"
        cumsum = np.cumsum(shapes)
        total = cumsum[-1]
        split1 = cumsum[0] 
        split2 = cumsum[-2]
        split3 = cumsum[-1]
        indx = np.array([0, split1, split2, split3])
        print(f"Using tripple split with splits at {split1} and {split2}, resulting in splits: {[split1, split2, total]}")

    if args.keep_n_groups is not None:
        n = args.keep_n_groups
        print(f"shapes {shapes}")
        assert n < len(shapes), f"Number of groups to keep together must be less than the total number of groups. Got {n} groups to keep together for {len(shapes)} total groups."
        print(f"Keeping the first {n}u groups together in one model.")
        cumsum = np.cumsum(shapes)
        indx = np.array([0] + [cumsum[n-1 ]])

    if args.global_coeff_norm:
        print("Using global coefficient normalization, computing global stats...")
        stats = compute_channel_stats(dataloader, base_transformer, coeff_normalize) if args.global_coeff_norm else None 
    else :
        stats = None

    transformers = []
    for i in range(len(indx) - 1):
        start = indx[i]
        end = indx[i+1]
        print(f"Creating transformer for slice {start}:{end}")
        transformer = get_transform(dataloader, transform_name=args.transform, input_example=sample, slice=(start,end), coeff_normalize=coeff_normalize, trunc=True , levels=levels, wavelet_type=wavelet_type, multi_level_coeff_normalize=coeff_multi_norm, stats=stats)
        transformers.append(transformer)

    coeff_bool = all(hasattr(transformer, "_denormalize") for transformer in transformers)
    if coeff_norm_requested and not coeff_bool:
        raise RuntimeError(
            "Coefficient normalization was requested, but no split transformer exposes denormalization."
        )

    if args.keep_n_groups is not None:
        assert len(transformers) == 1, f"When using --keep_n_groups, only one transformer should be created. Got {len(transformers)} transformers."
        base_transformer = get_transform(dataloader, transform_name=args.transform, input_example=sample, slice=(0,indx[1]), coeff_normalize=coeff_normalize, trunc=True , levels=levels, wavelet_type=wavelet_type)
    kept_coeffs = int(indx[-1])
    if args.drop_remainders and kept_coeffs < total_coeffs:
        print(f"Dropping remaining coefficients after split at {kept_coeffs}. Total coefficients before drop: {total_coeffs}, total coefficients after drop: {kept_coeffs}")
        base_transformer = get_transform(dataloader, transform_name=args.transform, input_example=sample, slice=(0,indx[-1]), coeff_normalize=coeff_normalize, trunc=True , levels=levels, wavelet_type=wavelet_type)


    def get_model_class(name):
        if name == "UnetODE":
            return UnetODE
        elif name == "TransformerODE":
            return TransformerODE
        elif name == "ChannelMLPODE":
            return ChannelMLPODE
        elif name == "MixerODE":
            return MixerODE
        elif name == "IdentityODE":
            return IdentityODE
        else: 
            return SimpleMLPODE
    
    transform_samples = [transformer.transform(sample) for transformer in transformers]

    get_model_class_ind = lambda index : get_model_class(args.model_class[index] if index < len(args.model_class) else args.model_class[-1])

    conditions = None
    models = []
    for i in range(len(transform_samples)):
        if cond == True:
            model = get_model_class_ind(i)(transform_samples[i], transform_samples[i-1] if i > 0 else None).to(device)
            conditions = [(lambda t : lambda x : t.transform(x))(transformer) for transformer in transformers[:-1]]
            conditions.insert(0, None) # add a None condition for the first model
        else:
            model = get_model_class_ind(i)(transform_samples[i]).to(device)
        models.append(model)

    if args.shaped_prior:
        idk = []
        for model, transform in zip(models, transformers):
            idk.append(ShapedPriorModel(model, transformer=transform, dataloader=dataloader))
        models = idk
    num_param = sum(p.numel() for model in models for p in model.parameters())
    config["num_param"] = num_param

    lossfunc = MSE()
    assert not (args.loss.startswith("new") and len(transformers) > 1), "The new_recon_loss and new_combined_loss are designed for single model training and cannot be used with multiple transformers. Please choose a different loss function or use only one transformer."
    lossfunc = get_loss_func(args.loss, transform=transformers[0], alpha=args.alpha)


    n = len(models)
    constr = "cond_" if cond == True else ""
    run_name = f'fm_{n}_stage_{constr}{window_size}{models[0].name}_{transformers[0].toString}_{dataset.toString()}_{lossfunc.toString}'

    run_dict = {
        "dataset": dataset.toString(),
        "window_size": window_size,
        "model_class": args.model_class,
        "transform": args.transform,
        "loss": args.loss,
        "alpha": args.alpha,
        "wavelet_type": wavelet_type,
        "levels": levels,
        "coeff_normalize": coeff_normalize,
        "multi_coeff_norm": coeff_multi_norm,
        "global_coeff_norm": args.global_coeff_norm,
        "splits": splits if splits is not None else "all",
        "no_split": no_split,
        "half_split": args.half_split,
        "level_split": args.level_split,
        "drop_remainders": args.drop_remainders,
        "tripple_split": args.tripple_split,
        "keep_n_groups": args.keep_n_groups,
        "disable_ot": args.disable_ot,
        "shaped_prior": args.shaped_prior,
        "eval": eval,
        "norm": norm,
        "cond": cond,
        "target_seq_len": args.target_seq_len,
        "model_names": [model.name for model in models],
    }

    config["run_dict"] = run_dict


    print(f'Training {run_name}')

    if args.logger == "mlflow":
        logger = MlFlowLogger(
            experiment_name="fm_experiment_freq_domain",
            run_name=run_name,
            config=config,
            mlflow_db=args.mlflow_db,
            mlflow_path=args.mlflow_path
        )
    elif args.logger == "wandb":
        logger = WandbLogger(
            experiment_name="fm_experiment_freq_domain",
            wandb_project="synth_time_series",
            wandb_entity="magnu-ww-ntnu",
            run_name=run_name,
            config=config,
            group=group
            # Add other WandbLogger-specific arguments if needed
        )
    else:
        logger = EmptyLogger()
    with logger.start_run():
        for model in models:
            logger.watch(model)

        print("Training Flow Matching")
        start_time = time.perf_counter()
        models, total_epochs = train_n_step_flow_matching_freq_domain(
            models=models, dataloader=dataloader, conditioning=conditions, config=config, transformers=transformers, device=device, logger=logger, lossfunc=lossfunc)
        training_time = time.perf_counter() - start_time
        logger.log_metrics(metrics={"training_time": training_time}, step=None)
        print("Sampling from trained model...") # samples_25 = sample_flow_freq_domain(model, transformer=transformer, latent_space_template=transform_sample, device=device, steps=25, n_samples=8)
        extract_and_store_samples(
            dataloader=dataloader,
            models=models,
            config=config,
            base_transformer=base_transformer,
            transformers=transformers,
            transform_samples=transform_samples,
            device=device,
            logger=logger,
            epoch=total_epochs,
            n_samples=100,
            cond=cond,
            show=False,
            steps=100,
            coeff_normalize=coeff_bool
        )
        print("Logging data...")
        log_data(
            models=models,
            dataloader=dataloader,
            base_transformer=base_transformer,
            transformers=transformers,
            transform_samples=transform_samples,
            logger=logger,
            device=device,
            cond=cond,
            n_samples=10000,
            steps=100,
            coeff_normalize=coeff_bool,
            artifact_folder=artifact_folder
        )
        print("Evaluating model...")
        eval_model(eval=eval,
                cond=cond,
                models=models,
                dataloader=dataloader,
                base_transformer=base_transformer,
                transformers=transformers,
                transform_samples=transform_samples,
                logger=logger,
                device=device,
                n_samples=len(dataloader.dataset),
                coeff_normalize=coeff_bool,
                steps=100)
        logger.log_source(models[0])
        logger.store_models(models, artifact_folder=artifact_folder)
        

def eval_model(
    eval,
    models,
    dataloader,
    base_transformer : Transform,
    transformers,
    transform_samples,
    logger : Logger,
    coeff_normalize : bool,
    device,
    cond=False,
    n_samples=10000,
    steps=10,):
    # for i in range(3):
    dataset = dataloader.dataset
    all_samples = DataLoader(dataset, batch_size=len(dataset))
    ori_data = next(iter(all_samples))[1]
    gen_coeffs = sample_nstep_flow_freq_domain(conditional=cond, models=models, transformers=transformers, latent_space_templates=transform_samples, device=device, steps=steps, n_samples=n_samples, coeff_normalize=coeff_normalize)
    gen_data = base_transformer.inverse_transform(gen_coeffs)
    metrics = evaluate_model_core(eval=eval, ori_data=ori_data, gen_data=gen_data)
    print("Evaluation results:", eval)
    logger.log_metrics(step=None, metrics=metrics)



def extract_and_store_samples(
    dataloader,
    models,
    config,
    base_transformer : Transform,
    transformers,
    transform_samples,
    device,
    logger : Logger,
    epoch,
    n_samples,
    coeff_normalize : bool,
    *,
    cond=False,
    plot_dist=True,
    plot_tsne=True,
    show=False,
    steps=25,
    return_intermidiates=True
):
    dataset = dataloader.dataset
    rand_sampler = RandomSampler(dataset, replacement=True, num_samples=n_samples)
    rand_loader = DataLoader(dataset, batch_size=1, sampler=rand_sampler)

    samples = []
    sample_coeffs = []
    for _, sample in rand_loader:
        coeff = base_transformer.transform(sample)
        samples.append(sample)
        # samples.append(sample)
        sample_coeffs.append(coeff)

    samples = torch.cat(samples, dim=0)
    sample_coeffs = torch.cat(sample_coeffs, dim=0)

    print("sampling")
    gen_coeffs = sample_nstep_flow_freq_domain(conditional=cond, models=models, transformers=transformers, latent_space_templates=transform_samples, device=device, steps=steps, step_size=1/steps, n_samples=n_samples, coeff_normalize=coeff_normalize, return_intermidiates=return_intermidiates)
    print("finshed sampling")

    sample_coeffs = format_coeffs_for_plotting(sample_coeffs, base_transformer)

    keys = config["dataset"]["y_key"]
    if show == True and return_intermidiates == True:
        for i in range(gen_coeffs[0].shape[0]):
            fig, ax = plt.subplots(4, 1, figsize=(8, 4))
            text_obj = fig.text(0.01, 0.95, "", fontsize=10, va='top')  # Add text at top-left
            cat_list = [coef[i:i+1,:,:] for coef in gen_coeffs]
            recs = [base_transformer.inverse_transform(i) for i in cat_list]
            def animate(timestep):
                # c1 = coeff_sample1[mask_nr]
                cat = format_coeffs_for_plotting(cat_list[timestep], base_transformer)
                rec = recs[timestep]
                actual = base_transformer.inverse_transform(sample_coeffs[i:i+1])
                actual_c = format_coeffs_for_plotting(sample_coeffs[i:i+1], base_transformer)

                for a in ax:
                    a.clear()
                plot_n_batches_vertical(actual, actual_c, cat, rec,axes=ax, fig=fig)  # Ensure your plot function accepts ax
                ax[2].set_title(f"timestep = {timestep}")
                ax[3].set_title(f"timestep = {timestep}")
            ani = FuncAnimation(fig, animate, frames=range(0, steps, 1), interval=20)
            plt.show()
        return
    if return_intermidiates:

        gen_samples = [base_transformer.inverse_transform(coeff) for coeff in gen_coeffs]
        _,c,_ = gen_coeffs[0].shape
        if len(keys) != c:
            raise ValueError(f"Expected number of channels in gen_coeffs to match the length of y_key, but got {c} and {len(keys)} respectively.")
        gen_coeffs = [format_coeffs_for_plotting(coeff, base_transformer) for coeff in gen_coeffs]
        for i, (gs,gc) in enumerate(zip(gen_samples, gen_coeffs)):
            if i % 50 == 0 or i == len(gen_coeffs) - 1:
                fig1 = plot_multiple_multivariate_samples_togheter_subfigure({"samples": samples, "sample_coeffs": sample_coeffs, "gen_coeffs": gc, "gen_samples": gs}, keys=keys)
                for k,v in fig1.items():
                    logger.store_plot(v, f'results_both_{k}', epoch +i)
                fig2 = plot_multiple_multivariate_samples_togheter_subfigure({"samples": samples, "sample_coeffs": sample_coeffs, "gen_coeffs": gc, "gen_samples": gs}, mode="lines", keys=keys)
                for k,v in fig2.items():
                    logger.store_plot(v, f'results_lines_{k}', epoch +i)
                fig3 = plot_multiple_multivariate_samples_togheter_subfigure({"samples": samples, "sample_coeffs": sample_coeffs, "gen_coeffs": gc, "gen_samples": gs}, mode="fan", keys=keys)
                for k,v in fig3.items():
                    logger.store_plot(v, f'results_fan_{k}', epoch +i)
        if plot_dist:
            for i in range(len(keys)):
                ori_data = samples[:,i:i+1,:].detach().cpu().numpy()
                gen_data = gen_samples[-1][:,i:i+1,:].detach().cpu().numpy()
                dist = plot_distribution(ori_data=ori_data, gen_data=gen_data)
                logger.store_plot(dist, f'results_distribution_{keys[i]}', None)
        if plot_tsne:
            ori_data = samples.detach().cpu().numpy()
            gen_data = gen_samples[-1].detach().cpu().numpy()
            dist = visualize_tsne(ori_data=ori_data, gen_data=gen_data)
            logger.store_plot(dist, f'results_tsne', None)
    else:
        gen_samples = base_transformer.inverse_transform(gen_coeffs)
        gen_coeffs = format_coeffs_for_plotting(gen_coeffs, base_transformer)
        fig = plot_multiple_samples_togheter_subfigure({"samples": samples, "sample_coeffs": sample_coeffs, "gen_coeffs": gen_coeffs, "gen_samples": gen_samples})
        logger.store_plot(fig, f'results', epoch)

    


def log_data(
    models,
    dataloader,
    base_transformer : Transform,
    coeff_normalize : bool,
    transformers,
    transform_samples,
    logger : Logger,
    device,
    artifact_folder,
    cond=False,
    n_samples=1000,
    steps=10,

):
    started = time.perf_counter()
    print("log_data: loading original samples", flush=True)
    dataset = dataloader.dataset
    all_samples = DataLoader(dataset, batch_size=len(dataset))
    ori_data = next(iter(all_samples))[1]

    print(f"log_data: loaded originals in {time.perf_counter() - started:.2f}s; generating {n_samples} samples", flush=True)
    step_started = time.perf_counter()
    gen_coeffs = sample_nstep_flow_freq_domain(conditional=cond, models=models, transformers=transformers, latent_space_templates=transform_samples, device=device, steps=steps, n_samples=n_samples, coeff_normalize=coeff_normalize)
    gen_coeff_time = time.perf_counter() - step_started
    print(f"log_data: generated coeffs in {gen_coeff_time:.2f}s; inverse transforming", flush=True)
    step_started = time.perf_counter()
    gen_samples = base_transformer.inverse_transform(gen_coeffs)

    print(f"log_data: inverse transformed in {time.perf_counter() - step_started:.2f}s; logging originals", flush=True)
    step_started = time.perf_counter()
    logger.log_data(
        artifact_folder=artifact_folder,
        file_name=f"original_samples",
        data=ori_data,
    )
    print(f"log_data: logged originals in {time.perf_counter() - step_started:.2f}s; logging generated samples", flush=True)
    step_started = time.perf_counter()
    logger.log_data(
        artifact_folder=artifact_folder,
        file_name=f"generated_samples",
        data=gen_samples,
        upload_data=False
    )
    print(f"log_data: logged generated samples in {time.perf_counter() - step_started:.2f}s; total {time.perf_counter() - started:.2f}s", flush=True)
    logger.log_metrics(metrics={"log_data_time": time.perf_counter() - started}, step=None)
    logger.log_metrics(metrics={"generated_samples_time": gen_coeff_time}, step=None)



if __name__ == "__main__":
    main()
