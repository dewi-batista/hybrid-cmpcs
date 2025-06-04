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
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None
    ):
        if k is not None:
            raise NotImplementedError

        logits_p = self.net(z)
        logits_p_chunks = tuple([logits_p]) if n_chunks is None else logits_p.chunk(n_chunks, dim=0)

        x_nan = x.clone()
        x_nan[:, -4:] = float('nan')

        log_prob_bins_numerator = torch.cat([bce_loss(logits_p_chunk, x, missing) for logits_p_chunk in logits_p_chunks], dim=1)
        log_prob_bins_denomintr = torch.cat([bce_loss(logits_p_chunk, x_nan, True) for logits_p_chunk in logits_p_chunks], dim=1)
        log_prob = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False) - torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
        return log_prob / 4

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
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None,
    ):
        z_chunks = tuple([z]) if n_chunks is None else z.chunk(n_chunks, dim=0)
        batch_size = x.shape[0]

        Y_num = 4 # if binary-hot encoded
        # Y_num = 1 # if one-hot encoded
        
        x_nan = x.detach().clone()
        x_nan[:, -Y_num:] = float('nan') 
        if k is not None:
            with torch.no_grad():
                # Run approximate posterior to find the 'best' k z values for each x
                log_prob_bins = torch.cat(
                    [ce_loss(self.net(z_chunk), x, k=None, missing=missing) - ce_loss(self.net(z_chunk), x_nan, k=None, missing=True) for z_chunk in z_chunks], dim=1
                )
                z_top_k = z[torch.topk(log_prob_bins, k=k, dim=-1)[1]] # shape (batch_size, k, latent_dim)
            
            log_prob_bins_top_k = ce_loss(self.net(z_top_k.view(batch_size * k, -1)), x, k=k, missing=missing)
            log_prob_bins_denomintr = ce_loss(self.net(z_top_k.view(batch_size * k, -1)), x_nan, k, missing=True)
            log_prob = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False) - torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
            return log_prob

        else:
            log_prob_bins_numerator = torch.cat([ce_loss(self.net(z_chunk), x, k, missing) for z_chunk in z_chunks], dim=1)
            log_prob_bins_denomintr = torch.cat([ce_loss(self.net(z_chunk), x_nan, k, True) for z_chunk in z_chunks], dim=1)
            log_prob = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False) - torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
            return log_prob

class ContinuousMixture(pl.LightningModule):

    def __init__(
        self,
        decoder: nn.Module,
        sampler: Callable,
        k: Optional[int] = None
    ):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.k = k
        self.n_chunks = None
        self.missing = None
        self.save_hyperparameters(ignore=[]), # ['n_chunks', 'missing', 'k']),
        self.val_losses = []

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        log_w: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None
    ):
        assert (z is None and log_w is None) or (z is not None and log_w is not None)
        if z is None:
            z, log_w = self.sampler(seed=seed)
        # log_prob_bins_numerator, log_prob_bins_denomintr = self.decoder.forward(x, log_w.to(x.device), z.to(x.device), k, self.missing, self.n_chunks)
        log_prob = self.decoder.forward(x, log_w.to(x.device), z.to(x.device), k, self.missing, self.n_chunks)
        # assert log_prob_bins_numerator.size() == (x.size(0), z.size(0)) or log_prob_bins_numerator.size() == (x.size(0), k)
        # log_prob_numerator = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False)
        # log_prob_denomintr = torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
        # return log_prob_numerator - log_prob_denomintr
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
        lls = torch.cat([self.forward(x.to(device), z, log_w, k=None, seed=seed) for x in loader], dim=0)
        assert len(lls) == len(loader.dataset)
        return lls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
