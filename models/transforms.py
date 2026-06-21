from typing import Protocol
import pywt
import ptwt
from scipy.sparse import rand
import torch
from torch.utils.data import DataLoader 
from torch import fft, Tensor, preserve_format, t, view_as_complex, view_as_real, nn
from abc import ABC, abstractmethod

from torch.nn.modules import transformer
from typing import Sequence, Literal, Tuple



class Transform(ABC):

    def __init__(self):
        self._repr_channels = 1
        self._toString = "Transform"
        self.has_custom_prior = False



    @property
    def repr_channels(self):
        return self.repr_channels

    @property
    def toString(self):
        return self._toString

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def inverse_transform(self, x: Tensor) -> Tensor:
        pass


class IdentityTransformer(Transform):
    def __init__(self):
        super().__init__()
        self._toString = "identity"
        self._repr_channels = 1

    def transform(self, x: Tensor) -> Tensor:
        return x

    def inverse_transform(self, x: Tensor | list[Tensor]) -> Tensor:
        if isinstance(x, list):
            raise TypeError(f"Identity transformer can only invert Tensors; got {type(x).__name__}")
        return x

class Fourier_transfomer(Transform):
    def __init__(self):
        super().__init__()
        self._toString = "fourier"
        self._repr_channels = 2

    def transform(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        real_tensor = view_as_real(fft.fft(x, norm="ortho"))
        idk = real_tensor.permute(0, 1, 3, 2).reshape(B, C * 2, L)
        return idk

    def inverse_transform(self, x: Tensor | list[Tensor]) -> Tensor:
        # return x
        if isinstance(x, list):
            raise TypeError(f"Fourier transformer can only invert Tensors; got {type(x).__name__}")
        B, C2, L = x.shape
        C = C2 // 2
        x = x.reshape(B, C, 2, L)
        x = x.permute(0, 1, 3, 2).contiguous()
        idk = fft.ifft(view_as_complex(x))
        return idk.real


class RFFTTransformer(Transform):

    def __init__(self, seq_length : int, norm : str = "forward") -> None:
        self.norm = norm
        self.length = seq_length
        self.shapes = (seq_length//2 + 1)
        self._toString = f'RFFT_{norm}'
        self._repr_channels = 2

    def transform(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        real_tensor = view_as_real(fft.rfft(x, norm=self.norm))
        idk = real_tensor.permute(0, 1, 3, 2)
        idk = idk.squeeze(1)
        return idk

    def inverse_transform(self, x: Tensor | list[Tensor]) -> Tensor:
        # return x
        if isinstance(x, list):
            raise TypeError(f"RFFT transformer can only invert Tensors; got {type(x).__name__}")
        if len(x.shape) == 4:
            b, f, c, l = x.shape
            assert l == (self.length//2 + 1), "This aint Hermitian"
            real = x[:,:,0,:]
            im = x[:,:,1,:]
        elif len(x.shape) ==3:
            b,c,l = x.shape
            assert l == (self.length//2 + 1), "This aint Hermitian"
            real = x[:,0,:]
            im = x[:,1,:]
        else:
            raise TypeError("Wrong x shape")
        half_complex = torch.complex(real,im)
        time = torch.fft.irfft(half_complex,n=self.length,norm=self.norm, dim=-1)
        if time.dim() == 2:
            time = time.unsqueeze(1)
        return time


class FlatWaveletTransformer(Transform):
    def __init__(self, wavelet_type : str, input_example : Tensor ,level :int,mode = "symmetric"):
        self.level = level 
        self.mode = mode
        self.wavelet = pywt.Wavelet(wavelet_type)
        self._toString = f'Flatwavelet_{wavelet_type}_level_{level}_mode_{mode}'
        coeffs = ptwt.wavedec(input_example, self.wavelet, mode=self.mode, level=self.level)
        self.shapes = [c.size(2) for c in coeffs]
        self.pywt = True


    def transform(self, x : Tensor) -> Tensor:
        coeff = ptwt.wavedec(x, self.wavelet, mode=self.mode, level=self.level)
        return torch.cat(coeff, dim=2) 
    def inverse_transform(self, x : Tensor) -> Tensor:
        splits = torch.split(x, self.shapes, dim=2)
        coeffs = list(splits)
        rec = ptwt.waverec(coeffs, wavelet=self.wavelet)
        return rec

class FlatPyWaveletTransformer(Transform):
    def __init__(self, wavelet_type: str, input_example: Tensor, level: int, mode="symmetric"):
        self.level = level
        self.mode = mode
        self.wavelet = pywt.Wavelet(wavelet_type)
        self._toString = f"FlatPyWavelet_{wavelet_type}_level_{level}_mode_{mode}"

        coeffs = pywt.wavedec(input_example.detach().cpu().numpy(), self.wavelet,
                              mode=self.mode, level=self.level, axis=2)
        self.shapes = [c.shape[2] for c in coeffs]
        self.original_length = input_example.size(2)
        self.pywt = True

    def transform(self, x: Tensor) -> Tensor:
        coeffs = pywt.wavedec(x.detach().cpu().numpy(), self.wavelet,
                              mode=self.mode, level=self.level, axis=2)
        coeffs = [torch.tensor(c, device=x.device, dtype=x.dtype) for c in coeffs]
        return torch.cat(coeffs, dim=2)

    def inverse_transform(self, x: Tensor) -> Tensor:
        coeffs = [s.detach().cpu().numpy() for s in torch.split(x, self.shapes, dim=2)]
        rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode, axis=2)
        return torch.tensor(rec[:, :, :self.original_length], device=x.device, dtype=x.dtype)

class TresholdingTransform(Transform):
    def __init__(self, base_transformer : Transform, treshold : float):
        super().__init__()
        self.base_transformer = base_transformer
        self._toString = f'Tresholding_{base_transformer.toString}_{treshold}'
        self.treshold = treshold

    def transform(self, x : Tensor, treshhold : float | None = None) -> Tensor:
        tresh = treshhold if treshhold is not None else self.treshold
        coeff = self.base_transformer.transform(x)
        coeff[coeff.abs() < tresh] = 0
        return coeff

    def inverse_transform(self, x : Tensor, treshold : float | None = None) -> Tensor:
        tresh = treshold if treshold is not None else self.treshold
        coeff = x
        coeff[coeff.abs() < tresh] = 0
        return self.base_transformer.inverse_transform(coeff)

def normalize(coeffs : Tensor, norm_type, norm_stats) -> Tensor:
    if norm_type in {"z", "minmax"} and len(norm_stats) != coeffs.shape[1]:
        raise ValueError(
            f"Expected {coeffs.shape[1]} channel stats, got {len(norm_stats)} for norm_type={norm_type}"
        )
    if norm_type == "z":
        idk = [(coeffs[:, c:c+1, :] - m) / (s + 1e-8) for c, (m, s) in enumerate(norm_stats)]
        return torch.cat(idk, dim=1)
    elif norm_type == "minmax":
        idk = [(coeffs[:, c:c+1, :] - mn) / ((mx - mn) + 1e-8) for c, (mn, mx) in enumerate(norm_stats)]
        return torch.cat(idk, dim=1)
    else:
        return coeffs
def denormalize(coeffs: Tensor, norm_type, norm_stats) -> Tensor:
    if norm_type in {"z", "minmax"} and len(norm_stats) != coeffs.shape[1]:
        raise ValueError(
            f"Expected {coeffs.shape[1]} channel stats, got {len(norm_stats)} for norm_type={norm_type}"
        )
    if norm_type == "z":
        idk = [(coeffs[:, c:c+1, :] * s) + m for c, (m, s) in enumerate(norm_stats)]
        return torch.cat(idk, dim=1)
    elif norm_type == "minmax":
        idk = [(coeffs[:, c:c+1, :] * ((mx - mn) + 1e-8)) + mn for c, (mn, mx) in enumerate(norm_stats)]
        return torch.cat(idk, dim=1)
    else:
        return coeffs

class NormalizedTransformer(Transform):
    def __init__(self, base_transformer : Transform, input_example : Tensor, norm_type : Literal["z", "minmax", None] = None, norm_stats : list[tuple[float,float]] | None = None):
        super().__init__()
        _, c, l = input_example.shape
        self.base_transformer = base_transformer
        self._toString = f'normalize_{base_transformer.toString}_{norm_type}'
        self.norm_type = norm_type
        self.norm_stats = norm_stats
        self.test(input_example)

    def transform(self, x : Tensor) -> Tensor:
        coeff = self.base_transformer.transform(x)
        normed_coeff = self._normalize(coeff)
        return normed_coeff

    def inverse_transform(self, x : Tensor) -> Tensor:
        denormed_coeffs = self._denormalize(x)
        rec = self.base_transformer.inverse_transform(denormed_coeffs)
        return rec

    def test(self, x : Tensor) -> Tensor:
        coeff = self.base_transformer.transform(x)
        normed_coeff = self._normalize(coeff)
        denormed_coeffs = self._denormalize(normed_coeff)
        if not torch.allclose(coeff, denormed_coeffs, atol=1e-5):
            print("Max abs diff:", (coeff - denormed_coeffs).abs().max().item())
            print("Input:", coeff)
            print("Reconstructed:", denormed_coeffs)
            raise AssertionError("Test failed: reconstructed signal does not match original")

    def _normalize(self, coeffs : Tensor) -> Tensor:
        return normalize(coeffs, self.norm_type, self.norm_stats)

    def _denormalize(self, coeffs: Tensor) -> Tensor:
        return denormalize(coeffs, self.norm_type, self.norm_stats)



class multi_level_NormalizedTransformer(Transform):
    def __init__(self, base_transformer : Transform, input_example : Tensor, norm_stats : list[list[tuple[float,float]]], norm_type : Literal["z", "minmax", None] = None):
        super().__init__()
        level_splits = list(level_splits)
        _, c, l = input_example.shape
        self.base_transformer = base_transformer
        self._toString = f'multi_level_normalize_{base_transformer.toString}_{norm_type}'
        self.norm_type = norm_type
        self.norm_stats =  norm_stats
        self.level_splits = level_splits
        assert len(level_splits) == len(self.norm_stats), f"Sum of level_splits {len(level_splits)} does not match length of norm_stats {len(self.norm_stats)}"

    def _normalize(self, coeffs : Tensor) -> Tensor:
        out = []
        for c, norm_stats in zip(torch.split(coeffs, self.level_splits, dim=2), self.norm_stats):
            out.append(normalize(c, self.norm_type, norm_stats))
        return torch.cat(out, dim=2)

    def _denormalize(self, coeffs: Tensor) -> Tensor:
        out =[]
        for c, norm_stats in zip(torch.split(coeffs, self.level_splits, dim=2), self.norm_stats):
            out.append(denormalize(c, self.norm_type, norm_stats))
        return torch.cat(out, dim=2)

    def transform(self, x : Tensor) -> Tensor:
        coeff = self.base_transformer.transform(x)
        return self._normalize(coeff)
    def inverse_transform(self, x : Tensor) -> Tensor:
        denormed_coeffs = self._denormalize(x)
        return self.base_transformer.inverse_transform(denormed_coeffs)

class automaticMaskTransform(Transform):
    def __init__(self, transformer: Transform, input_example: Tensor):
        super().__init__()
        self._toString = f'AutomaticMask_{transformer.toString}'
        sample = transformer.transform(input_example)
        self.seq_length = sample.shape[2]
        self.base_transformer = transformer

    def transform(self, x: Tensor) -> Tensor:
        coeff = self.base_transformer.transform(x)
        return coeff 
    def inverse_transform(self, x: Tensor) -> Tensor:
        B, C, L = x.shape

        coeff = torch.zeros(
            B,
            C,
            self.seq_length,
            dtype=x.dtype,
            device=x.device,
        )

        coeff[:, :, :L] = x

        return self.base_transformer.inverse_transform(coeff)

class SimpleMaskTransform(Transform):
    def __init__(self, transformer: Transform, input_example: Tensor, trunc = False, mask_nr : int | None = None, slice : Tuple[int, int] | None = None):
        super().__init__()
        if mask_nr is not None and slice is not None:
            raise ValueError("Cannot specify both mask_nr and slice parameters.")

        if mask_nr is None and slice is None:
            raise ValueError("Must specify either mask_nr or slice parameter.")

        split = mask_nr if mask_nr is not None else slice[1]
        self._toString = f'SimpleMask_{transformer.toString}_{split}'
        sample = transformer.transform(input_example)
        self.seq_length = sample.shape[2]
        self.mask_nr = mask_nr
        self.base_transformer = transformer
        self.trunc = trunc
        if slice is not None:
            if slice[1] > self.seq_length:
                raise ValueError(f"Slice end {slice[1]} exceeds sequence length {self.seq_length}")
        self.slice = slice

    def transform(self, x: Tensor) -> Tensor:
        coeff = self.base_transformer.transform(x)
        if self.trunc:
            if self.slice is not None:
                coeff = coeff[:, :, self.slice[0]:self.slice[1]]
            else:
                coeff = coeff[:, :, :self.mask_nr]
        else:
            if self.slice is not None:
                mask = torch.ones_like(coeff)
                mask[:, :, self.slice[0]:self.slice[1]] = 0
                coeff = coeff * mask
            else:
                coeff[:, :, self.mask_nr:] = 0
        return coeff 

    def inverse_transform(self, x: Tensor) -> Tensor:
        if self.trunc:
            if self.slice is not None:
                if self.slice[1] > self.seq_length:
                    raise ValueError(f"Slice end {self.slice[1]} exceeds sequence length {self.seq_length}")
                padded = torch.zeros(
                    x.shape[0], x.shape[1], self.seq_length, 
                    dtype=x.dtype, device=x.device
                )
                padded[:, :, self.slice[0]:self.slice[1]] = x
            else:
                padded = torch.zeros(
                    x.shape[0], x.shape[1], self.seq_length, 
                    dtype=x.dtype, device=x.device
                )
                padded[:, :, :self.mask_nr] = x
            return self.base_transformer.inverse_transform(padded)
        else:
            return self.base_transformer.inverse_transform(x)


def most_sig_coef(coeff, nr_coeffs : int):
    score = torch.linalg.norm(coeff, axis=1, keepdims=True)

    topk_vals, topk_idx = torch.topk(score, nr_coeffs, dim=-1)  # (B, k)
    index_counts = torch.bincount(topk_idx.flatten())
    most_common_indices = torch.topk(index_counts, nr_coeffs).indices
   


    mask = torch.zeros_like(score)  # (B, L)
    mask.scatter_(-1, topk_idx, 1.0)
    out = coeff * mask

    return out

class TopKMaskTransform(Transform):
    def __init__(self, transformer: Transform, input_example: Tensor, mask_nr : int):
        super().__init__()
        self._toString = f'TopKMask_{transformer.toString}_{mask_nr}'
        sample = transformer.transform(input_example)
        self.seq_length = sample.shape[2]
        self.mask_nr = mask_nr
        self.base_transformer = transformer

    def transform(self, x: Tensor) -> Tensor:
        coeffs = self.base_transformer.transform(x)
        masked_coeff = most_sig_coef(coeffs, self.mask_nr)
        return masked_coeff

    def inverse_transform(self, x: Tensor) -> Tensor:
        return self.base_transformer.inverse_transform(x)


#### THIS IS NOT A DATA TRANSFORM, IT IS JUST A HACK TO EASILY IMPLEMENT CUSTOM PRIORS IN THE COEFFICIENT SPACE.

class HeuristicTopKMaskTransform(Transform):
    def __init__(self, transformer: Transform, dataloader, input_example: Tensor, nr_coeff : int):
        super().__init__()
        self._toString = f'HeuristicTopKMask_{transformer.toString}_{nr_coeff}'
        sample = transformer.transform(input_example)
        self.seq_length = sample.shape[2]
        self.nr_coeff = nr_coeff
        self.base_transformer = transformer
        self.most_common_topk_indicies = self.find_most_common_topk_indices(dataloader)

    def find_most_common_topk_indices(self, Datal) -> Tensor:
        idk = []
        for _, x in Datal:
            idk.append(x)
        x = torch.cat(idk, dim=0)
        coeff = self.base_transformer.transform(x)
        score = torch.linalg.norm(coeff, axis=1, keepdims=True)
        topk_vals, topk_idx = torch.topk(score, self.nr_coeff, dim=-1)  # (B, k)
        index_counts = torch.bincount(topk_idx.flatten())
        most_common_indices = torch.topk(index_counts,  self.nr_coeff).indices
        return most_common_indices


    def transform(self, x: Tensor) -> Tensor:
        coeffs = self.base_transformer.transform(x)
        return coeffs[:, :, self.most_common_topk_indicies]

    def inverse_transform(self, x: Tensor) -> Tensor:
        # pad the masked coefficients with zeros to restore original shape
        padded = torch.zeros(
            x.shape[0], x.shape[1], self.seq_length, 
            dtype=x.dtype, device=x.device
        )
        padded[:, :, self.most_common_topk_indicies] = x
        return self.base_transformer.inverse_transform(padded)

def estimate_coeff_magnitude(transformer, dataloader, nr_coeff):
    magnitudes = []
    for _, y in dataloader:
        # Apply the transform
        coeffs = transformer.transform(y)
        coeffs_of_importance = coeffs[:, :, nr_coeff:]  
        # Compute mean absolute value per sample, then average over batch
        batch_magnitude = coeffs.abs().mean().item()
        magnitudes.append(batch_magnitude)
    # Return the average magnitude over all processed batches
    return sum(magnitudes) / len(magnitudes)


def estimate_coeff_stats(transformer, dataloader, nr_coeff):
    coeffs_list = []
    for _, y in dataloader:
        coeffs = transformer.transform(y)
        coeffs_of_importance = coeffs[:, :, nr_coeff:]  # shape: (b, 1, l)
        coeffs_list.append(coeffs_of_importance)
    # Concatenate along batch dimension
    all_coeffs = torch.cat(coeffs_list, dim=0)  # shape: (total_b, 1, l)
    # Compute mean and std across batches (dim=0)
    mean = all_coeffs.mean(dim=0).squeeze(0)  # shape: (l,)
    std = all_coeffs.std(dim=0).squeeze(0)    # shape: (l,)
    return mean, std

def sample_coeffs_from_stats(mean, std, batch_size, narrowing_factor = 1.0):
    # mean, std: shape (l,)
    l = mean.shape[0]
    mean = mean.view(1, 1, l)  # shape: (1, 1, l)
    std = std.view(1, 1, l)    # shape: (1, 1, l)
    std = std * narrowing_factor
    samples = torch.normal(mean.expand(batch_size, 1, l), std.expand(batch_size, 1, l))
    return samples  # shape: (b, 1, l)

class Noised_simplemask(nn.Module, Transform):
    def __init__(self, base_transformer: Transform, dataloader, input_example: Tensor, nr_coeff : int, narrowing_factor = 1.0):
        super().__init__()
        self.example = base_transformer.transform(input_example)
        self.example_shape = base_transformer.transform(input_example).shape
        self.nr_coeff = nr_coeff
        self.narrowing_factor = narrowing_factor
        self._toString = f'NoisedsimpleMask_{base_transformer.toString}_{nr_coeff}'
        self.base_transformer = base_transformer
        self.top_transform = SimpleMaskTransform(base_transformer,input_example, True, nr_coeff)
        self.stats = estimate_coeff_stats(base_transformer, dataloader, nr_coeff)


    def transform(self, x: Tensor) -> Tensor:
        out = torch.zeros(self.example_shape)
        coeffs = self.top_transform.transform(x)
        rand_coeffs = sample_coeffs_from_stats(self.stats[0], self.stats[1], x.shape[0], self.narrowing_factor)
        out[:, :, :self.nr_coeff] = coeffs
        out[:, :, self.nr_coeff:] = rand_coeffs
        return out

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_transformer.inverse_transform(x)



