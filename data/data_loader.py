import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import scipy.io.arff as arff
import numpy as np
from typing import TypedDict, Literal, List, Tuple

from models.transforms import Transform

def sine_gen(data_points: int = 128,
             amplitude: tuple[float, float] = (1.0, 1.0),
             length: tuple[float, float] = (2.0, 2.0)):
    amp = np.random.uniform(amplitude[0], amplitude[1])
    leng = np.random.uniform(length[0], length[1])
    leng = leng * 2 * np.pi
    x = np.linspace(0, leng, num=data_points, endpoint=False)
    sine = amp * np.sin(x)
    return torch.tensor(sine, dtype=torch.float32).unsqueeze(0)


class SineDatasetDict(TypedDict):
    name: Literal["SINE"]
    size: int
    window_size: int
    amplitude: Tuple[int, int]
    length: Tuple[int, int]

class SineDataset(Dataset):
    def __init__(self, config : SineDatasetDict):
        super().__init__()
        self.name = "SINE"
        self.size = config["size"] 
        self.data_points = config["window_size"] 
        self.amplitude = config["amplitude"] 
        self.length = config["length"] 

    def __len__(self):
        return self.size

    def toString(self):
        return "SINE"

    def __getitem__(self, idx):
        return [] ,sine_gen(
            data_points=self.data_points,
            amplitude=self.amplitude,
            length=self.length
        )


def get_sine_loader(n: int = 50,
                    data_points: int = 128,
                    amplitude: tuple[float, float] = (1.0, 1.0),
                    length: tuple[float, float] = (2.0, 2.0),
                    batch_size: int = 8,
                    shuffle: bool = True):
    dataset = SineDataset(n, data_points, amplitude, length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



class HDF5Dataset(Dataset):
    def __init__(self, h5_path: str,
                 feature_keys: list[str],
                 y_key: str,
                 num_samples: int
                 ):
        super().__init__()
        self.h5_path = h5_path
        self.num_samples = num_samples
        df = pd.read_hdf(h5_path)
        self.length = len(df)
        self.features = df[feature_keys]
        self.y = df[y_key]

    def __len__(self):
        return self.length - self.num_samples + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()          # converts 0-d tensor -> Python int
        else:
            idx = int(idx)
        x_np = self.features.iloc[idx:idx+self.num_samples].values
        y_np = self.y.iloc[idx:idx+self.num_samples].values
        # -> [num_features, num_samples]
        x = torch.tensor(x_np, dtype=torch.float32).T
        # -> [num_samples]
        y = torch.tensor(y_np, dtype=torch.float32).squeeze(-1)
        y = y.unsqueeze(0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RICODatasetDict(TypedDict):
    h5_path: str
    feature_keys: List[str]
    y_key: List[str]
    window_size: int
    phase_mode: Literal["heating", "cooling", "both"]
    name : Literal["RICO"]


class RICODataset(Dataset):
    def __init__(self, config: RICODatasetDict):
        super().__init__()
        h5_path = config["h5_path"]
        feature_keys = config["feature_keys"]
        y_key = config["y_key"]
        window_size = config["window_size"]


        self.name = f'rico_{window_size}'
        self.window_size = window_size

        df = pd.read_hdf(h5_path)


        self.features = df[feature_keys]
        self.y = df[y_key[0]]
        self.length = len(df)
        self.valid_indices = []

        for i in range(self.length - window_size + 1):
            window_phase = self.phase.iloc[i:i+window_size]

            if (window_phase == window_phase.iloc[0]).all():
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def toString(self) -> str:
        return "RICO"
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size

        x_np = self.features.iloc[start_idx:end_idx].values
        y_np = self.y.iloc[start_idx:end_idx].values

        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)
        if len(self.features) > 0:
            x = torch.tensor(x_np, dtype=torch.float32).T
            return x,y

        return None, y




class FILEDatasetDict(TypedDict):
    name: Literal["FILE"]
    dataset_name : str
    fileFormat : Literal["CSV","ARFF", "HD5"]
    filepath: str
    y_key: List[str]
    window_size: int
    dropna: bool
    normalize:  None | Literal["MINMAX", "Z"]
    remove_outliers_std: float | None
    transform : Transform | None

class FILEDataset(Dataset):

    def __init__(self, config: FILEDatasetDict):
        # Expect a config TypedDict with keys declared in CSVDatasetDict
        super().__init__()
        self.name = config["name"]
        self.dataset_name = config["dataset_name"]
        self.filepath = config["filepath"]
        self.y_key = config["y_key"]
        self.num_samples = int(config["window_size"])
        self.dropna = config["dropna"]
        self.normalize = config["normalize"].upper() if config["normalize"] is not None else None
        # Will hold normalization statistics per column (e.g. min/max or mean/std)
        self.norm_stats = None
        self.remove_outliers_std = config["remove_outliers_std"]
        self.transform = config["transform"]

        match config["fileFormat"]:
            case "ARFF":
                data, _ = arff.loadarff(self.filepath)
                df = pd.DataFrame(data)
            case "CSV":
                df = pd.read_csv(self.filepath)
        if self.dropna:
            df = df.dropna().reset_index(drop=True)



        if self.remove_outliers_std is not None:
            n_before = len(df)
            combined_mask = pd.Series(False, index=df.index)
            for col in self.y_key:
                if col not in df.columns:
                    continue
                series = pd.to_numeric(df[col], errors="coerce")
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or np.isnan(std):
                    continue
                outlier_mask = (series < mean - self.remove_outliers_std * std) | (series > mean + self.remove_outliers_std * std)
                combined_mask = combined_mask | outlier_mask
            df = df[~combined_mask].reset_index(drop=True)
            n_after = len(df)
            print(f"n_before {n_before} n_after {n_after} Removed {n_before - n_after} outliers")

        if self.normalize is not None:
            self.norm_stats = self._compute_norm_stats(df)
            self._apply_normalization(df, self.norm_stats)

        self.length = len(df)
        self.y = df[self.y_key]
        self._create_windows()


    def _compute_norm_stats(self, df):
        stats = {}
        for col in self.y_key:
            if col not in df.columns:
                continue
            series = pd.Series(pd.to_numeric(df[col], errors="coerce"))
            if self.normalize == "MINMAX":
                col_min = float(series.min())
                col_max = float(series.max())
                stats[col] = {"min": col_min, "max": col_max}
            elif self.normalize == "Z":
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                stats[col] = {"mean": mean, "std": std}
        print(stats)
        return stats

    def _apply_normalization(self, df, stats):
        for col in self.y_key:
            if col not in df.columns:
                continue
            series = pd.Series(pd.to_numeric(df[col], errors="coerce"))
            if self.normalize == "MINMAX":
                col_min = stats[col]["min"]
                col_max = stats[col]["max"]
                if col_max - col_min == 0:
                    norm_series = series - col_min
                else:
                    norm_series = (series - col_min) / (col_max - col_min)
                df[col] = norm_series
            elif self.normalize == "Z":
                mean = stats[col]["mean"]
                std = stats[col]["std"]
                if std == 0 or np.isnan(std):
                    norm_series = series - mean
                else:
                    norm_series = (series - mean) / std
                df[col] = norm_series
            elif self.normalize == "softz":
                std = stats[col]["std"]
                assert std is not None, "std cannot be None for Soft_z normalization"
                assert std >= 0, "std cannot be negative for Soft_z normalization"
                norm_series = series / std 
                df[col] = norm_series

    def _create_windows(self):
        windows = []
        for i in range(self.length - self.num_samples + 1):
            window_y = self.y.iloc[i:i+self.num_samples].values
            windows.append(window_y)
        self.windows = windows


    def __len__(self):
        # number of valid windows
        return max(0, self.length - self.num_samples + 1)

    def toString(self) -> str:
        return f"{self.dataset_name}_{self.normalize}"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        idx = int(idx)
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError('index out of range')

        y = self.windows[idx]
        y = torch.tensor(y, dtype=torch.float32).T
        return [], y

    # def __getitem__(self, idx):
    #     # support tensor and negative indexing
    #     if torch.is_tensor(idx):
    #         idx = idx.item()
    #     idx = int(idx)
    #     if idx < 0:
    #         idx = len(self) + idx
    #
    #     if idx < 0 or idx >= len(self):
    #         raise IndexError('index out of range')
    #
    #     start = idx
    #     end = start + self.num_samples
    #
    #     y_np = self.y.iloc[start:end].values         
    #
    #     y = torch.tensor(y_np, dtype=torch.float32).T  
    #     return [], y


def plot_features_and_y(features: torch.Tensor, y: torch.Tensor, feature_names=None, y_name='y'):
    features_np = features.squeeze().detach().cpu().numpy()
    y_np = y.squeeze().detach().cpu().numpy()
    num_samples = features_np.shape[1]
    x_axis = range(num_samples)
    plt.figure(figsize=(10, 6))
    for i in range(features_np.shape[0]):
        label = feature_names[i] if feature_names is not None else f'feature_{
            i}'
        plt.plot(x_axis, features_np[i, :], label=label)
    plt.plot(x_axis, y_np, label=y_name, linestyle='--', color='black')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Features and y')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_dataset(datasetConfig: SineDatasetDict | FILEDatasetDict | RICODatasetDict) -> SineDataset | RICODataset | FILEDataset:
    match datasetConfig["name"]:
        case "SINE":
            return SineDataset(datasetConfig)
        case "RICO":
            return RICODataset(datasetConfig)
        case "FILE":
            return FILEDataset(datasetConfig)

