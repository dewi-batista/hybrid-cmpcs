from torch.utils.data import DataLoader
from torch.distributions import Normal
from typing import Callable, Optional
from utils.losses import bce_loss, mse_loss, ce_loss
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np

class BernoulliDecoder(nn.Module):

    def __init__(
        self,
        net: nn.Module
    ):
        super(BernoulliDecoder, self).__init__()
        self.net = net

    def forward(
        self,
        x: torch.Tensor,
        log_w: torch.Tensor,
        z: torch.Tensor,
        lamda,
        X_num,
        Y_num,
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None,
    ):
        if k is not None:
            raise NotImplementedError

        logits_p = self.net(z)
        logits_p_chunks = tuple([logits_p]) if n_chunks is None else logits_p.chunk(n_chunks, dim=0)

        bce_image = torch.cat([bce_loss(logits_p_chunk[:, :-4], x[:, :-4], missing=False) for logits_p_chunk in logits_p_chunks], dim=1)
        bce_label = torch.cat([bce_loss(logits_p_chunk[:, -4:], x[:, -4:], missing=False) for logits_p_chunk in logits_p_chunks], dim=1)
        
        log_prob_bins_numerator = bce_image + bce_label
        log_prob_bins_denomintr = bce_image
        
        log_prob_joint = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False)
        log_prob_margi = torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
        
        weight_numerator = lamda / Y_num + (1 - lamda) / (X_num + Y_num)
        weight_denomintr = lamda / Y_num
        
        log_prob = weight_numerator * log_prob_joint - weight_denomintr * log_prob_margi

        return log_prob

class CategoricalDecoder(nn.Module):

    def __init__(
            self,
            net: nn.Module):
        super(CategoricalDecoder, self).__init__()
        self.net = net

    def forward(
        self,
        x: torch.Tensor,
        log_w: torch.Tensor,
        z: torch.Tensor,
        lamda,
        X_num,
        Y_num,
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None,
    ):
        z_chunks = tuple([z]) if n_chunks is None else z.chunk(n_chunks, dim=0)
        batch_size = x.shape[0]
        x_nan = x.detach().clone()
        x_nan[:, -4:] = float('nan') # change to -4 if bin-hot encoded
        
        z_chunk = z_chunks[0]
        print(self.net(z_chunk).shape)
        print(x[:, :-4].shape)

        ce_image = torch.cat([ce_loss(self.net(z_chunk)[:, :-4], x[:, :-4], k, False) for z_chunk in z_chunks], dim=1)
        ce_label = torch.cat([ce_loss(self.net(z_chunk)[:, -4:], x[:, -4:], k, False) for z_chunk in z_chunks], dim=1)
        
        log_prob_bins_numerator = ce_image + ce_label
        log_prob_bins_denomintr = ce_image
        
        log_prob_joint = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False)
        log_prob_margi = torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
        
        weight_numerator = lamda / Y_num + (1 - lamda) / (X_num + Y_num)
        weight_denomintr = lamda / Y_num
        
        log_prob = weight_numerator * log_prob_joint - weight_denomintr * log_prob_margi

        return log_prob

class ContinuousMixture(pl.LightningModule):

    def __init__(
        self,
        decoder: nn.Module,
        sampler: Callable,
        lamda: float,
        X_num: int,
        Y_num: int,
        k: Optional[int] = None
    ):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.k = k
        self.lamda = lamda
        self.X_num = X_num
        self.Y_num = Y_num
        self.n_chunks = None
        self.missing = None
        self.save_hyperparameters(ignore=[]),
        self.val_losses = []

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        log_w: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None
    ):
        if (z is None and log_w is None):
            z, log_w = self.sampler(seed=seed)
        
        log_prob = self.decoder.forward(
            x, log_w.to(x.device), z.to(x.device), self.lamda, self.X_num, self.Y_num, k, self.missing, self.n_chunks
        )
        return log_prob

    def training_step(
        self,
        x: torch.Tensor,
        batch_idx: int
    ):
        log_prob = self.forward(x, k=self.k)
        loss = (-log_prob).mean()
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, x: torch.Tensor, batch_idx: int):
        log_prob = self.forward(x, k=None, seed=42)
        loss = (-log_prob).mean()
        self.val_losses.append(loss.detach())
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('valid_loss_epoch', avg_loss, prog_bar=True, logger=True)
        self.val_losses.clear()

    @torch.no_grad()
    def eval_loader(
        self,
        loader: DataLoader,
        z: Optional[torch.Tensor] = None,
        log_w: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = False,
        device: str = 'cuda'
    ):
        self.eval()
        loader = tqdm(loader) if progress_bar else loader
        # lls = torch.cat([self.forward(x.to(device), z, log_w, k=None, seed=seed) for x in loader], dim=0)
        
        # only for test time! it uses lambda=0, X_num=0 and Y_num=1 so it returns NLL
        lls = torch.cat([
            self.decoder.forward(
                x.to(device), log_w.to(device), z.to(device), 0, 0, 1, self.k, self.missing, self.n_chunks
            )
            for x in loader],
        dim=0)
        
        assert len(lls) == len(loader.dataset)
        return lls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
