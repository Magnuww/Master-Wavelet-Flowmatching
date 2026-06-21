from typing import Literal, Tuple
from torch import Tensor
from utils.stats import compute_channel_stats, compute_channel_stats_per_split
from models.transforms import IdentityTransformer, RFFTTransformer, FlatWaveletTransformer, SimpleMaskTransform, HeuristicTopKMaskTransform, NormalizedTransformer, multi_level_NormalizedTransformer, TresholdingTransform, Transform

def get_transform(
    dataloader,
    transform_name: Literal["identity", "rfft", "flatwavelet"],
    input_example: Tensor,
    firstk : int | None = None,
    slice : Tuple[int, int] | None = None,
    topk: int | None = None,
    trunc: bool = False,
    wavelet_type: str = "db4",
    levels: int = 3,
    coeff_normalize: Literal["z", "minmax", None] = None,
    multi_level_coeff_normalize: None | list[int] = None,
    stats = None,
    tresh : float | None = None
) -> Transform:
    transform_name_lower = transform_name.lower()
    transformer = IdentityTransformer()
    if transform_name_lower == "rfft":
        print("Using RFFT transformer")
        transformer = RFFTTransformer(seq_length=input_example.shape[2])
    if transform_name_lower == "flatwavelet":
        print("Using flatwavelet transformer")
        transformer = FlatWaveletTransformer(wavelet_type=wavelet_type, input_example=input_example, level=levels)
    if firstk is not None or slice is not None:
        print(f"Using SimpleMaskTransform with firstk={firstk} and slice={slice}")
        transformer = SimpleMaskTransform(transformer=transformer, input_example=input_example, slice=slice, trunc=trunc)
    elif topk is not None:
        transformer = HeuristicTopKMaskTransform(transformer=transformer, input_example=input_example, dataloader=dataloader, nr_coeff=topk)
    if coeff_normalize is not None:
        if multi_level_coeff_normalize is not None:
            print(f"Using multi_level_NormalizedTransformer with norm_type={coeff_normalize} and level_splits={multi_level_coeff_normalize}")
            split_channel_stats = compute_channel_stats_per_split(dataloader, transformer, multi_level_coeff_normalize, coeff_normalize)
            transformer = multi_level_NormalizedTransformer(base_transformer=transformer, input_example=input_example, norm_type=coeff_normalize, norm_stats=split_channel_stats)
        else:
            print(f"Using NormalizedTransformer with norm_type={coeff_normalize} and stats={stats}")
            if stats is None:
                stats = compute_channel_stats(transformer=transformer, dataloader=dataloader, type=coeff_normalize)
            transformer = NormalizedTransformer(base_transformer=transformer, input_example=input_example, norm_type=coeff_normalize, norm_stats=stats)
    if tresh is not None:
        print(f"Using TresholdingTransform with treshold={tresh}")
        transformer = TresholdingTransform(base_transformer=transformer, treshold=tresh)
    return transformer
