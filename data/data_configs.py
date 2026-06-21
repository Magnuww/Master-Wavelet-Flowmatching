from models.transforms import Transform
from .data_loader import RICODatasetDict, FILEDatasetDict, get_dataset
from typing import Literal, List
def get_data_config(
        dataset_name: Literal["RICO", "google_stock", "etth1", "eeg", "SINE"],
        window_size: int,
        normalize: Literal["MINMAX", "Z", None] = None,
        remove_outliers_std: float | None = None,
        transformer : Transform | None = None,
        y_key : List[str] | None = None) -> RICODatasetDict | FILEDatasetDict:
    dataset_name_lower = dataset_name.lower()
    dataset = None
    if dataset_name_lower == "rico":
        ricodataset: RICODatasetDict = {
            "name": "RICO",
            "window_size": window_size,
            "feature_keys": [],
            "y_key": "B.RTD1",
            "h5_path": "./datasets/RICO_Acquisition_1_07-2023.hdf",
            "phase_mode": "cooling"
        }
        dataset = ricodataset
    elif dataset_name_lower == "google_stock":
        googlstock: FILEDatasetDict = {
            "name": "FILE",
            "dataset_name": "google_stock",
            "window_size": window_size,
            "filepath": "./datasets/GOOGL.csv",
            "fileFormat": "CSV",
            "y_key": ["Open","High","Low","Close","Adj Close","Volume"],
            # "y_key": ["Volume"],
            "dropna": True,
            "normalize": normalize,
            "remove_outliers_std": None,
            "transform": transformer
        }
        dataset = googlstock
    elif dataset_name_lower == "etth1":
        etth1: FILEDatasetDict = {
            "name": "FILE",
            "dataset_name": "etth1",
            "window_size": window_size,
            "filepath": "./datasets/ETTh1.csv",
            "fileFormat": "CSV",

            "y_key": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"],
            # "y_key": ["HUFL",],
            "dropna": True,
            "normalize": normalize,
            "remove_outliers_std": None,
            "transform": transformer
        }
        dataset = etth1
    elif dataset_name_lower == "eeg":
        eeg: FILEDatasetDict = {
            "name": "FILE",
            "dataset_name": "EEG_eye_state",
            "window_size": window_size,
            "filepath": "./datasets/EEG Eye State.arff",
            "fileFormat": "ARFF",
            # "y_key": ["AF3"],
            "y_key": ["AF3", "F7" , "F3" , "FC5", "T7" , "P7" , "O1" , "O2" , "P8" , "T8" , "FC6", "F4" , "F8" , "AF4"],
            "dropna": True,
            "normalize": normalize,
            "remove_outliers_std": 4.5,
            "transform": transformer
        }
        dataset = eeg

    # elif dataset_name_lower == "sine":
    #     sinedataset: SineDatasetDict = {
    #         "name": "SINE",
    #         "window_size": window_size,
    #         "amplitude": (1, 1),
    #         "length": (1, 1),
    #         "size": 500,
    #     }
    #     dataset = sinedataset
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name_lower}")
    if y_key is not None and dataset is not None:
        dataset["y_key"] = y_key
    return dataset
