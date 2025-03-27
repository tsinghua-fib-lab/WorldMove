import inspect
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dataset.feature import EMB_MAX, EMB_MEAN, EMB_MIN, EMB_STD
from metrics.metrics import (
    complete_transition_matrix,
    daily_loc,
    duration,
    gyration_radius,
    ks_test,
    travel_distance,
)
from modules.unet import Guide_UNet as BaselineUNet
from modules.utils import get_grid_num, get_region_emb, weight_init
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

P_MEAN = -1.2
P_STD = 1.2


class RegionDiff(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super(RegionDiff, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg["model"]
        self.dataset = cfg["dataset"]["name"]
        self.norm_data = cfg["dataset"]["norm"]
        self.target = self.cfg["target"]
        self.noise_prior = self.cfg["noise_prior"]
        self.input_dim = self.cfg["input_dim"]
        self.output_dim = self.cfg["output_dim"]
        self.lr = self.cfg["lr"]
        self.lr_scheduler = self.cfg["lr_scheduler"]
        self.weight_decay = self.cfg["weight_decay"]
        self.T_max = self.cfg["T_max"]
        self.metrics = self.cfg["metrics"]
        self.diffusion_steps = self.cfg["diffusion"]["num_steps"]
        self.sample_steps = self.cfg["diffusion"]["num_sample_steps"]
        self.beta_start = self.cfg["diffusion"]["beta_start"]
        self.beta_end = self.cfg["diffusion"]["beta_end"]

        # diffusion params
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.diffusion_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.unet_decoder = BaselineUNet(cfg=cfg)

        # region embedding
        self.region_emb = torch.from_numpy(
            get_region_emb(cfg["dataset"]["name"])
        ).float()
        self.grid_num = get_grid_num(cfg["dataset"]["name"])

        # validation metrics
        self.valildation_step_outputs = {
            m: {"real": [], "gen": []} for m in self.metrics
        }
        self.valildation_step_outputs["transition"] = {
            "real": torch.zeros(
                (
                    self.grid_num["x"] * self.grid_num["y"],
                    self.grid_num["x"] * self.grid_num["y"],
                )
            ),
            "gen": torch.zeros(
                (
                    self.grid_num["x"] * self.grid_num["y"],
                    self.grid_num["x"] * self.grid_num["y"],
                )
            ),
        }

        self.apply(weight_init)

    def forward(
        self, data: torch.Tensor, noised_gt: torch.Tensor, noise_label: torch.Tensor
    ) -> torch.Tensor:
        noised_gt = noised_gt.to(torch.float32)
        out = self.unet_decoder(data, noised_gt, noise_label)
        return out

    def training_step(self, batch, batch_idx):
        # add noise to gt
        gt = self._get_training_target(batch)
        batch_size = gt.shape[0]
        t = torch.randint(
            low=0, high=self.diffusion_steps, size=(batch_size // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.diffusion_steps - t - 1], dim=0)[:batch_size]
        c = self.alpha_bar.to(gt.device).gather(-1, t).reshape(-1, 1, 1)
        mean = c**0.5 * gt
        var = 1 - c
        eps = torch.randn_like(gt).to(gt.device)
        xt = mean + (var**0.5) * eps

        # forward
        out = self.forward(batch, xt, t)

        # loss
        loss = F.mse_loss(eps.float(), out)

        self.log(
            "train_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        real_idx = self._reconstruct_idx(self._get_training_target(batch))
        sample = self.sampling(batch, num_steps=self.sample_steps, show_progress=False)
        gen_idx = self._reconstruct_idx(sample)
        # metrics
        if "distance" in self.metrics:
            self.valildation_step_outputs["distance"]["real"].extend(
                travel_distance(real_idx, self.grid_num)
            )
            self.valildation_step_outputs["distance"]["gen"].extend(
                travel_distance(gen_idx, self.grid_num)
            )
        if "radius" in self.metrics:
            self.valildation_step_outputs["radius"]["real"].extend(
                gyration_radius(real_idx, self.grid_num)
            )
            self.valildation_step_outputs["radius"]["gen"].extend(
                gyration_radius(gen_idx, self.grid_num)
            )
        if "duration" in self.metrics:
            self.valildation_step_outputs["duration"]["real"].extend(
                duration(real_idx, self.grid_num)
            )
            self.valildation_step_outputs["duration"]["gen"].extend(
                duration(gen_idx, self.grid_num)
            )
        if "daily_loc" in self.metrics:
            self.valildation_step_outputs["daily_loc"]["real"].extend(
                daily_loc(real_idx, self.grid_num)
            )
            self.valildation_step_outputs["daily_loc"]["gen"].extend(
                daily_loc(gen_idx, self.grid_num)
            )
        if "cpc" in self.metrics or "mape" in self.metrics:
            self.valildation_step_outputs["transition"][
                "real"
            ] += complete_transition_matrix(real_idx, self.grid_num)
            self.valildation_step_outputs["transition"][
                "gen"
            ] += complete_transition_matrix(gen_idx, self.grid_num)

    def on_validation_epoch_end(self):
        if "distance" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["distance"]["real"],
                self.valildation_step_outputs["distance"]["gen"],
            )
            if len(gen) == 0:
                gen = [0]
            ks_stat = ks_test(real, gen)
            self.log(
                "distance_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["distance"]["real"].clear()
            self.valildation_step_outputs["distance"]["gen"].clear()
        if "radius" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["radius"]["real"],
                self.valildation_step_outputs["radius"]["gen"],
            )
            ks_stat = ks_test(real, gen)
            self.log(
                "radius_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["radius"]["real"].clear()
            self.valildation_step_outputs["radius"]["gen"].clear()
        if "duration" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["duration"]["real"],
                self.valildation_step_outputs["duration"]["gen"],
            )
            if len(gen) == 0:
                gen = [0]
            ks_stat = ks_test(real, gen)
            self.log(
                "duration_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["duration"]["real"].clear()
            self.valildation_step_outputs["duration"]["gen"].clear()
        if "daily_loc" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["daily_loc"]["real"],
                self.valildation_step_outputs["daily_loc"]["gen"],
            )
            if len(gen) == 0:
                gen = [0]
            ks_stat = ks_test(real, gen)
            self.log(
                "daily_loc_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["daily_loc"]["real"].clear()
            self.valildation_step_outputs["daily_loc"]["gen"].clear()
        if "cpc" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["transition"]["real"],
                self.valildation_step_outputs["transition"]["gen"],
            )
            cpc = (2 * torch.sum(torch.min(real, gen)) / torch.sum(real + gen)).item()
            self.log(
                "cpc",
                value=cpc,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
        self.valildation_step_outputs["transition"]["real"].zero_()
        self.valildation_step_outputs["transition"]["gen"].zero_()

    def _get_training_target(self, batch) -> torch.Tensor:
        return batch[..., : self.input_dim]

    def _reconstruct_idx(self, sample: torch.Tensor) -> torch.Tensor:
        if self.target == "emb":
            sample = sample.to(self.region_emb.device)
            if self.norm_data:
                # sample = sample * EMB_STD + EMB_MEAN
                sample = (sample + 1) / 2 * (EMB_MAX - EMB_MIN) + EMB_MIN
            sample = sample.float()
            distance = torch.cdist(
                sample,
                self.region_emb[..., : self.output_dim],
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            sample_idx = torch.argmin(distance, dim=-1)
            return sample_idx
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sampling(
        self,
        data: Mapping[str, torch.Tensor],
        latent: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        eta: float = 0.0,
        return_his: bool = False,
        show_progress: bool = True,
    ) -> torch.Tensor | List[torch.Tensor]:
        if return_his:
            res = []
        device = next(self.parameters()).device
        # latent
        batch_size, seq_len, _ = self._get_training_target(data).shape
        output_dim = self.output_dim
        if latent is None:
            latent = torch.randn((batch_size, seq_len, output_dim), dtype=torch.float64)
            if self.noise_prior:
                latent = latent * self.noise_std[None, :, None]
                # latent = self.noise_sampling(data)
        latent = latent.to(device)
        # denoising time steps
        t_steps = range(0, self.diffusion_steps, self.diffusion_steps // num_steps)
        t_next = [-1] + list(t_steps[:-1])
        beta = torch.cat([torch.zeros(1).to(device), self.beta.to(device)], dim=0).to(
            torch.float64
        )
        alpha_cumprod = (1 - beta).cumprod(dim=0)

        x_next = latent
        if show_progress:
            bar = tqdm.tqdm(
                total=num_steps, unit="step", desc="Sampling", dynamic_ncols=True
            )
        for t_c, t_n in zip(reversed(t_steps), reversed(t_next)):
            t_cur = torch.ones((batch_size,), dtype=torch.long, device=device) * t_c
            t_next = torch.ones((batch_size,), dtype=torch.long, device=device) * t_n
            pre_noise = self.forward(data, x_next, t_cur).to(torch.float64)

            at = alpha_cumprod.index_select(0, t_cur + 1).view(-1, 1, 1)
            at_next = alpha_cumprod.index_select(0, t_next + 1).view(-1, 1, 1)

            x0_t = (x_next - pre_noise * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = (1 - at_next - c1**2).sqrt()
            eps = torch.randn(x_next.shape, device=x_next.device)
            x_next = at_next.sqrt() * x0_t + c1 * eps + c2 * pre_noise

            if return_his:
                res.append(x_next.cpu().numpy())
            if show_progress:
                bar.update(1)
        if show_progress:
            bar.close()
        if return_his:
            return res
        return x_next

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.LSTMCell,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.Embedding,
            nn.GroupNorm,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        params = {
            "optimizer": optimizer,
            "T_max": self.T_max,
            "total_steps": self.T_max,
            "max_lr": self.lr,
            "pct_start": 0.15,
        }
        s = eval(self.lr_scheduler)
        scheduler = s(
            **{k: v for k, v in params.items() if k in inspect.signature(s).parameters}
        )
        return [optimizer], [scheduler]
