import torch.nn as nn
import torch

def get_decoder_debd(
    latent_dim: int,
    out_features: int,
    n_layers: int,
    batch_norm: bool,
    final_act=None
):
    hidden_dims = torch.arange(latent_dim, out_features, (out_features - latent_dim) / n_layers, dtype=torch.int)
    decoder = nn.Sequential()
    for i in range(len(hidden_dims) - 1):
        decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        decoder.append(nn.LeakyReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(hidden_dims[i + 1]))
    decoder.append(nn.Linear(hidden_dims[-1], out_features))
    if final_act is not None:
        decoder.append(final_act)
    return decoder
