from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver, ODESolver
from flow_matching.utils import ModelWrapper
from pydantic import InstanceOf
from models.models import IdentityODE, ShapedPriorModel
from models.optimal_transport import get_ot_plan, sample_ot_minibatch
from models.transforms import Transform, RFFTTransformer
from sklearn.utils.validation import _num_samples
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from models.lossfuncs import New_Combined_loss, New_Recon_loss
from tqdm import tqdm
from utils.logger import Logger
from utils.plot_utils import plot_multiple_samples_togheter_subfigure, format_coeffs_for_plotting
from typing import Callable
import numpy as np
import torch
from metrics.batch_metrics import compute_metrics


class composit_dataset(torch.utils.data.Dataset):
    def __init__(self, ori :Tensor, transformer, conditioning : Callable[[Tensor], Tensor] | None = None, device="cuda"):
        self.dataset = ori.to(device)
        self.transformed_data = transformer.transform(ori).to(device)
        
        if conditioning is not None:
            self.conditioning_data = conditioning(ori).to(device)
        else:
            self.conditioning_data = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.conditioning_data is not None:
            return self.dataset[idx], self.transformed_data[idx], self.conditioning_data[idx]
        return self.dataset[idx], self.transformed_data[idx]

def train_flow_matching_freq_domain(model : nn.Module, dataloader : DataLoader, transformer : Transform, config, epochs : int, lossfunc : nn.Module | None = None, device="cuda",logger : Logger | None =None, logging_name="", conditioning : None | Callable[[Tensor], Tensor] = None, epoch_offset = 0):
    if isinstance(model, IdentityODE):
        print("Model is IdentityODE, skipping training.")
        return model

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.train()

    #ugly fix to avoid doing transforms each loop, 2x model speedup
    dataset = dataloader.dataset
    all_samples = DataLoader(dataset, batch_size=len(dataset))
    idk = next(iter(all_samples))[1]
    print(f"Original data shape: {idk.shape}")
    c = composit_dataset(idk, transformer, conditioning)

    new_dataloader = DataLoader(c, batch_size=config["batch_size"], shuffle=True)
    #
    path = AffineProbPath(scheduler=CondOTScheduler())
    latent_space_template = None 
    # epochs = config["epochs"]
    for epoch in range(epochs):
        epoch += epoch_offset #offset to make logging across models in wandb work better
        model.train()
        total_loss = 0.0
        for batch in tqdm(new_dataloader):
            if conditioning is not None:
                ori, x_1, cond = batch
            else:
                ori, x_1 = batch
                cond = None

            if latent_space_template is None:
                 latent_space_template = x_1

            x_1 = x_1.to(device)
            if isinstance(model, ShapedPriorModel):
                x_0 = model.sample_prior_like(x_1, device=device)
            else:
                x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0), device=device)  
            if not config["disable_ot"]: 
                x_0 = sample_ot_minibatch(x_0, x_1 )
            path_sample = path.sample(x_0,x_1,t)

            if cond is not None:
                if config["cond_noise_dropout"] != None:
                    std = config["cond_noise_dropout"]["std"]
                    dropout = config["cond_noise_dropout"]["dropout"]
                    if torch.rand(1).item() < dropout:
                        cond = torch.zeros_like(cond)
                    else:
                        cond = cond + std * torch.randn_like(cond)

                    cond = cond.to(device)
                else:
                    cond = cond.to(device)
                v_pred = model(path_sample.x_t, path_sample.t, cond)
            else:
                v_pred = model(path_sample.x_t, path_sample.t)
            assert (
                path_sample.dx_t.shape == v_pred.shape
            ), f"Shape mismatch: path_sample.dx_t.shape={path_sample.dx_t.shape} != v_pred.shape={v_pred.shape}"

            if lossfunc is not None:
                if isinstance(lossfunc, New_Recon_loss) or isinstance(lossfunc, New_Combined_loss):
                    loss = lossfunc(v_pred, ori, path_sample)
                else:
                    loss = lossfunc(v_pred, path_sample.dx_t)
            else:
                loss = ((v_pred - path_sample.dx_t) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(
            f"epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")
        if logger:
            logger.log_metrics(step=epoch, metrics={f"{logging_name}loss":avg_loss})
        if logger != None and latent_space_template != None:
            if config["log_per_epoch"] != None and epoch % config["log_per_epoch"] == 0 :
                coeffs = sample_flow_freq_domain(model, transformer=None, device=device, latent_space_template=latent_space_template, n_samples=1000)
                samples = transformer.inverse_transform(coeffs)
                coeffs = format_coeffs_for_plotting(coeffs, transformer)

                labels = config["dataset"]["y_key"]
                b, c, l = coeffs.shape
                if len(labels) != c:
                    raise ValueError(f"Expected number of channels in coeffs to match the length of y_key, but got {c} and {idk} respectively.")
                for i in range(len(labels)):
                    c = coeffs[:,i:i+1,:]
                    s = samples[:,i:i+1,:]
                    fig = plot_multiple_samples_togheter_subfigure({f"{logging_name}gen_coeffs": c, f"{logging_name}gen_samples": s})
                    logger.store_plot(fig, f'figures/epochs/coeffs_and_samples_{logging_name}{labels[i]}', step=epoch)
            if config["eval_per_epoch"] != None and epoch % config["eval_per_epoch"] == 0 :
                metrics = evaluate_model(eval=config["during_training_eval_metrics"], model=model, dataloader=dataloader, transformer=transformer, latent_space_template=latent_space_template, device=device, steps=25, n_samples=1000, step_size=0.05)
                logger.log_metrics(step=epoch, metrics=metrics)
    if logger and config["log_model"] == True:
        logger.store_model(model)
    return model

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        c = extras.get("c", None)
        if c is not None:
            return self.model(x, t, c)
        else:
            return self.model(x, t)

@torch.no_grad()
def sample_flow_freq_domain(model, transformer : Transform | None, latent_space_template : Tensor, device="cuda", steps=25, return_intermidiates=False, n_samples=1, step_size=0.05, conditions : Tensor | None = None):
    model.eval()
    if conditions is not None:
        print(f"Sampling with conditions of shape {conditions.shape}")
        assert isinstance(conditions, Tensor), f"conditions must be a Tensor, got {type(conditions).__name__}"
        conditions = conditions.to(device)
        b = conditions.shape[0]
        assert b == n_samples, f"Batch size of conditions must match n_samples, but got {b} and {n_samples} respectively."

    wrapped_vf = WrappedModel(model)
    solver = ODESolver(wrapped_vf)
    T = torch.linspace(0,1,steps)  # sample times
    T = T.to(device=device)
    assert n_samples > 0, "n_samples has to be greater than 0"
    sample = torch.zeros((n_samples, *latent_space_template.shape[1:]), device=device)
    if isinstance(model, ShapedPriorModel):
        x_init = model.sample_prior_like(sample, device=device)
    else:
        x_init = torch.randn_like(sample)


    sample_params={
        "time_grid": T,
        "x_init": x_init,
        "method": 'dopri5',
        "step_size": None,
        "return_intermediates": return_intermidiates,
    }
    if conditions is not None:
        sample_params["c"] = conditions
    sol = solver.sample(**sample_params) 
    
    if not isinstance(sol, Tensor):
        raise TypeError(f"sol must be a Tensor, got {type(sol).__name__}")
    return sol

def train_n_step_flow_matching_freq_domain(models : list[nn.Module],
                                           dataloader : DataLoader,
                                           transformers : list[Transform],
                                           config,
                                           lossfunc : nn.Module | None = None,
                                           device="cuda",
                                           logger : Logger | None =None,
                                           logging_name="",
                                           conditioning : list[Callable[[Tensor], Tensor] | None] | None  = None):
    out = []
    for model in models:
        model.to(device)
    cond = None
    if conditioning is not None:
        assert len(conditioning) == len(models), f"Length of conditioning list must match the number of models, but got {len(conditioning)} and {len(models)} respectively."
    if conditioning is None:
        conditioning = [None] * len(models)
    epochs = config["epochs"]
    get_epcoh = lambda i: epochs[i] if i < len(epochs) else epochs[-1]
    epoch_offset = 0
    for i, (model, transformer, cond) in enumerate(zip(models, transformers, conditioning)):
        print(f"Training model {i+1}/{len(models)}")
        epoch = get_epcoh(i)
        model = train_flow_matching_freq_domain(model=model, dataloader=dataloader, transformer=transformer, config=config, epochs=epoch, lossfunc=lossfunc, device=device, logger=logger, logging_name=f"{logging_name}_model{i}_", conditioning=cond, epoch_offset=epoch_offset)
        epoch_offset += epoch
        out.append(model)
    return out, epoch_offset




def sample_batch_nstep_flow_freq_domain(models, transformers, latent_space_templates : Tensor, device="cuda", steps=25, return_intermidiates=False, n_samples=1, step_size=0.05, conditional : bool = False, coeff_normalize = False):
    all_coeffs = []
    conds = None
    for i, (model, latent_space_template) in enumerate(zip(models, latent_space_templates)):
        coeffs = sample_flow_freq_domain(model=model, transformer=None, latent_space_template=latent_space_template, device=device, steps=steps, return_intermidiates=return_intermidiates, n_samples=n_samples, step_size=step_size, conditions=conds)
        all_coeffs.append(coeffs)
        if conditional:
            conds = coeffs[-1] if return_intermidiates else coeffs

    #Closure that ensures that the correct transformer is bound to the lambda function
    #Lambda functions responsible for denormalizing after transform if coeff_normalize is True
    #Necessary becuase the transformers normalize differently and the need to have the same scale
    denom_funcs = [
        (lambda t: (lambda x: t._denormalize(x) if coeff_normalize else x))(transformer)
        for transformer in transformers
    ]

    if return_intermidiates:
        denom_coeff = []
        for coeffs, denom in zip(all_coeffs, denom_funcs):
            denom_coeff.append([denom(c) for c in coeffs])
        return [torch.cat(coeffs, dim=2) for coeffs in zip(*denom_coeff)]
    else:
        return torch.cat([denom_func(coeffs) for denom_func, coeffs in zip(denom_funcs, all_coeffs)], dim=2)

def sample_nstep_flow_freq_domain(models, transformers, latent_space_templates : Tensor, device="cuda", steps=25, return_intermidiates=False, n_samples=1, step_size=0.05, conditional : bool = False, coeff_normalize = False, batch_split_length = 1000):
    if n_samples <= batch_split_length:
            return sample_batch_nstep_flow_freq_domain(
                models=models,
                transformers=transformers,
                latent_space_templates=latent_space_templates,
                device=device,
                steps=steps,
                return_intermidiates=return_intermidiates,
                n_samples=n_samples,
                step_size=step_size,
                conditional=conditional,
                coeff_normalize=coeff_normalize,
            )
    all_samples = []

    for start in range(0, n_samples, batch_split_length):
        current_batch_size = min(batch_split_length, n_samples - start)

        batch_samples = sample_batch_nstep_flow_freq_domain(
            models=models,
            transformers=transformers,
            latent_space_templates=latent_space_templates,
            device=device,
            steps=steps,
            return_intermidiates=return_intermidiates,
            n_samples=current_batch_size,
            step_size=step_size,
            conditional=conditional,
            coeff_normalize=coeff_normalize,
        )
        if return_intermidiates:
            batch_samples = [sample.cpu() for sample in batch_samples]
        else:
            batch_samples = batch_samples.cpu()

        all_samples.append(batch_samples)

    if return_intermidiates:
        return [
            torch.cat(samples_at_step, dim=0)
            for samples_at_step in zip(*all_samples)
        ]

    return torch.cat(all_samples, dim=0)



def evaluate_model(eval : dict | None,  model, dataloader : DataLoader, transformer : Transform | None, latent_space_template : Tensor, device="cuda", steps=25, n_samples=1000, step_size=0.05):
    if eval is None:
        return {}
    dataset = dataloader.dataset
    all_samples = DataLoader(dataset, batch_size=len(dataset))
    ori_data = next(iter(all_samples))[1]
    gen_data = sample_flow_freq_domain(model=model, transformer=transformer, latent_space_template=latent_space_template, device=device, steps=steps, n_samples=n_samples, step_size=step_size)
    if transformer is not None:
        gen_data = transformer.inverse_transform(gen_data)
    return evaluate_model_core(eval=eval, ori_data=ori_data, gen_data=gen_data)

def evaluate_model_core(eval : dict | None, ori_data, gen_data, full_results=False):
    if eval is None:
        return {}
    assert isinstance(gen_data, Tensor), f"gen_data must be a Tensor, got {type(gen_data).__name__}"
    ori_data = ori_data.detach().permute(0,2,1).cpu().numpy()
    gen_data = gen_data.detach().permute(0,2,1).cpu().numpy()
    scores = compute_metrics(ori_data, gen_data, metrics=eval["method_list"], return_full_results=full_results)
    return scores

