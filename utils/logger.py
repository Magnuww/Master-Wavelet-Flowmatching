from abc import ABC
from datetime import datetime
from contextlib import contextmanager
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import json
from typing import Dict, Self, Iterator, Any, Literal
import wandb
import torch
import os
import numpy as np

class Logger(ABC):
    def log_metrics(self, step: int | None, metrics: dict[str , Any]):
        pass

    def store_plot(self, figure, artifact_path, step):
        pass

    @contextmanager
    def start_run(self) -> Iterator[Self]:
        # Default context manager that yields self (no-op run)
        yield self

    def watch(self, model, log="all"):
        pass
    def store_models(self, models, optimizer=None, artifact_folder : str | None = None, epoch=None):
        pass

    def load_into_models(self, models, artifact_folder="runs", exp_name="default_exp", device="cpu"):
        pass

    def store_model(self, model: Any, optimizer = None, export_model: bool = False):
        pass
    def log_dataloader_simple(self, dataloader: DataLoader, context: str = "training"):
        pass
    def log_params(self, params : dict[Any, Any]):
        pass
    def log_data(self, data: Any, file_name: str, artifact_folder: str | None = None, upload_data: bool = True):
        pass

    def log_source(self, model):
        pass

class EmptyLogger(Logger):
    """A no-op logger that implements the Logger interface but does nothing."""

    def log_metrics(self, step: int | None, metrics: dict[str , Any]) -> None:
        return None

    def store_plot(self, figure: Any, artifact_path: str, step : int) -> None:
        return None
    def watch(self, model, log="all"):
        return None
    def log_source(self, model):
        return None

    @contextmanager
    def start_run(self) -> Iterator[Self]:
        yield self

    def log_data(self, data: Any, file_name: str, artifact_folder: str | None = None, upload_data: bool = True):
        return None
    def store_models(self, models, optimizer=None, artifact_folder : str | None = None, epoch=None):
        return None
    def load_into_models(self, models, artifact_folder="runs", exp_name="default_exp", device="cpu"):
        return None

    def store_model(self, model: Any, optimizer = None, export_model: bool = False) -> None:
        return None
    def log_dataloader_simple(self, dataloader: DataLoader, context: str = "training"):
        return None
    def log_params(self, params : dict[Any, Any]):
        return None

class MlFlowLogger(Logger):

    def __init__(self, experiment_name, run_name, config : Dict, mlflow_path=None, mlflow_db=None):
        self.cfg = dict(config)

        if mlflow_path:
            mlflow.set_tracking_uri(f'File:/{mlflow_path}')
        if mlflow_db:
            mlflow.set_tracking_uri(f'sqlite:///{mlflow_db}')
        mlflow.set_experiment(experiment_name)

        # Start run with nice naming
        self.run_name = f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

        # Log ALL config values as params (except experiment name)
        self.params_to_log = {k: v for k, v in self.cfg.items() if k != "experiment"}


    @contextmanager
    def start_run(self):
        mlflow.pytorch.autolog()
        with mlflow.start_run(run_name=self.run_name) as run:
            mlflow.log_params(self.params_to_log)
            print("Run logging started:", run.info.run_id)
            yield self
        print("Run logging exited")

    def log_metrics(self, step: int | None, metrics: dict[str, Any]):
        mlflow.log_metrics(metrics, step=step)

    def watch(self, model, log="all"):
        return None
            

    def log_params(self, params : dict[Any, Any]):
        mlflow.log_params(params=params)

    def store_plot(self , figure, artifact_path, step=None):

        if step is not None:
            if artifact_path.endswith('.png'):
                artifact_path = artifact_path[:-4] + f"_step{step}.png"
            else:
                artifact_path = f"{artifact_path}_step{step}.png"
        elif not artifact_path.endswith('.png'):
            artifact_path = f"{artifact_path}.png"
        mlflow.log_figure(figure,artifact_path)

    def store_model(self,model, optimizer=None, export_model=False):
        mlflow.pytorch.log_model(model,export_model=export_model)

    def log_dataloader_simple(self, dataloader: DataLoader, context: str = "training"):
        mlflow.log_params({
            f"{context}_n_samples":   len(dataloader.dataset),
            f"{context}_batch_size":  dataloader.batch_size,
            f"{context}_n_batches":   len(dataloader),
        })


class WandbLogger(Logger):

    def __init__(self, experiment_name, run_name, config: Dict, wandb_project=None, wandb_entity=None, group=None):
        self.cfg = dict(config)
        self.experiment_name = experiment_name
        self.run_name = f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
        self.wandb_project = wandb_project or experiment_name
        self.wandb_entity = wandb_entity
        self.wandb_run = None

        # Log ALL config values as params (except experiment name)
        self.params_to_log = {k: v for k, v in self.cfg.items() if k != "experiment"}
        self.group = group

    @contextmanager
    def start_run(self):
        wandb_args = {
            "project": self.wandb_project,
            "entity": self.wandb_entity,
            "name": self.run_name,
            "config": self.params_to_log,
            "reinit": True,
        }
        if self.group is not None:
            wandb_args["group"] = self.group
        self.wandb_run = wandb.init(**wandb_args)
        print("Run logging started:", self.wandb_run.id)
        try:
            yield self
        finally:
            wandb.finish()
            print("Run logging exited")

    def log_metrics(self, step: int | None, metrics: dict[str, Any]):
        wandb.log(metrics, step=step)

    def watch(self, model, log = "all"):
        if log not in ["all", "gradients", "parameters"]:
             raise ValueError(f"Invalid log option: {log}. Must be 'all', 'gradients', or 'parameters'.")
        wandb.watch(model, log=log)

    def log_params(self, params: dict[Any, Any]):
        wandb.config.update(params, allow_val_change=True)

    def store_plot(self, figure, artifact_path, step):
        artifact_path = artifact_path if artifact_path.endswith('.png') else f"{artifact_path}.png"
        wandb.log({artifact_path: wandb.Image(figure)},step=step)

    def log_data(self, data: torch.Tensor, file_name: str, artifact_folder: str | None = None, upload_data: bool = True):
        if artifact_folder is None:
            artifact_folder = "runs"
        exp_name = self.wandb_run.name if self.wandb_run else "default_exp"
        if exp_name is None:
            exp_name = "default_exp"
        path = os.path.join(artifact_folder, exp_name)
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{file_name}.npy")

        np.save(save_path, data.cpu().numpy())

        if upload_data:
            self.wandb_run.save(save_path)

    def store_models(self, models, optimizer=None, artifact_folder : str | None = None, epoch=None):
        if artifact_folder is None:
            artifact_folder = "runs"
        exp_name = self.wandb_run.name if self.wandb_run else "default_exp"
        if exp_name is None:
            exp_name = "default_exp"
        path = os.path.join(artifact_folder, exp_name)
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(models):
            checkpoint = {
                "model_state_dict": model.state_dict(),
            }
            save_path = os.path.join(path, f"model_{i}.ckpt")
            torch.save(checkpoint, save_path)
            artifact = wandb.Artifact(f"model_{i}.ckpt", type="model")
            artifact.add_file(save_path)
            self.wandb_run.log_artifact(artifact)

    def load_into_models(self, models, artifact_folder="runs", exp_name="default_exp", device="cpu"):
        path = os.path.join(artifact_folder, exp_name)
        for i, model in enumerate(models):
            checkpoint_path = os.path.join(path, f"model_{i}.ckpt")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

        return models


    def log_source(self, model):
        # Log the source code of the model's class
        import inspect
        source_code = inspect.getsource(model.__class__)
        artifact = wandb.Artifact("model_source_code", type="code")
        with artifact.new_file("model_source.py") as f:
            f.write(source_code)
        self.wandb_run.log_artifact(artifact)

    def log_dataloader_simple(self, dataloader: DataLoader, context: str = "training"):
        wandb.config.update({
            f"{context}_n_samples": len(dataloader.dataset),
            f"{context}_batch_size": dataloader.batch_size,
            f"{context}_n_batches": len(dataloader),
        }, allow_val_change=True)
