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
        
        # # print(logits_p.shape, x.shape)
        # # logits_p: [B*k, 784], x: [B, 784]
        # D = x.size(1)               # == 784
        # # build one Boolean mask of length 784: randomly drop 20% of the first 780, keep last 4
        # mask = torch.cat([
        #     torch.rand(D-4, device=x.device) > 0.2,     # keep ~80% of cols 0…779
        #     torch.ones(4,  device=x.device, dtype=torch.bool)  # always keep cols 780–783
        # ])
        # # apply the same mask to both:
        # x        = x[:,        mask]   # now [B, ~628]
        # logits_p = logits_p[:, mask]   # now [B*k, ~628]
        # # print(logits_p.shape, x.shape)

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

    def test_time(
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
        
        log_prob_joint = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False)
        
        # weight_numerator = lamda / Y_num + (1 - lamda) / (X_num + Y_num)
        # weight_denomintr = lamda / Y_num
        
        # log_prob = weight_numerator * log_prob_joint - weight_denomintr * log_prob_margi

        return log_prob_joint

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
        x_nan = x.detach().clone()
        x_nan[:, -1:] = float('nan') # change to -4 if bin-hot encoded
        if k is not None:
            print("I better not be in use!!")
            with torch.no_grad():
                # Run approximate posterior to find the 'best' k z values for each x
                log_prob_bins = torch.cat(
                    [ce_loss(self.net(z_chunk), x, k=None, missing=missing) - ce_loss(self.net(z_chunk), x_nan, k=None, missing=True) for z_chunk in z_chunks], dim=1)
                # log_prob_bins = log_prob_bins + log_w.unsqueeze(0)
                z_top_k = z[torch.topk(log_prob_bins, k=k, dim=-1)[1]]  # shape (batch_size, k, latent_dim)
            
            log_prob_bins_top_k = ce_loss(self.net(z_top_k.view(batch_size * k, -1)), x, k=k, missing=missing)
            log_prob_bins_denomintr = ce_loss(self.net(z_top_k.view(batch_size * k, -1)), x_nan, k, missing=True)
            return log_prob_bins_top_k, log_prob_bins_denomintr

        else:
            log_prob_bins_numerator = torch.cat([ce_loss(self.net(z_chunk), x, k, missing) for z_chunk in z_chunks], dim=1)
            log_prob_bins_denomintr = torch.cat([ce_loss(self.net(z_chunk), x_nan, k, True) for z_chunk in z_chunks], dim=1)
            # log_prob = torch.logsumexp(log_prob_bins_numerator + log_w.unsqueeze(0), dim=1, keepdim=False) - torch.logsumexp(log_prob_bins_denomintr + log_w.unsqueeze(0), dim=1, keepdim=False)
            
            # check that this give the same!!
            log_prob = torch.logsumexp(log_prob_bins_numerator, dim=1, keepdim=False) - torch.logsumexp(log_prob_bins_denomintr, dim=1, keepdim=False)
            # print(log_prob_bins_numerator[0][0])
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
        
        # log_prob = self.decoder.test_time(
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
        lls = torch.cat([self.forward(x.to(device), z, log_w, k=None, seed=seed) for x in loader], dim=0)
        assert len(lls) == len(loader.dataset)
        return lls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
