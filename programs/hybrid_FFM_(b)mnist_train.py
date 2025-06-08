#=== START directory shenanigans

import os
import sys

# 1. repo_dir used later
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. sys.path must be appended for importing modules
sys.path.append(repo_dir)

# 3. fix current working directory
programs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
os.chdir(programs_dir)

#=== END directory shenanigans

from models.cm_hybrid import BernoulliDecoder, CategoricalDecoder, ContinuousMixture
from models.nets import get_decoder_debd, mnist_conv_decoder
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.bins_samplers import GaussianQMCSampler
from utils.reproducibility import seed_everything

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch

# choose to use MNIST or binarised MNIST
use_mnist = False
dset = 'mnist' if use_mnist else 'bmnist'

# check if we're running with CUDA-utilising hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = None if device == 'cpu' else 1

# create data directory (if needed)
data_dir = "../data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# download MNIST into data directory (if needed)
mnist_train_and_val = datasets.MNIST(root="../data", train=True, download=True)

# assign labels
labels_mnist_train_and_val = mnist_train_and_val.targets

# convert datasets to tensors
mnist_train_and_val = mnist_train_and_val.data.view(60_000, 784).float()

# embed class label in final pixel(s) of training samples
for idx in range(mnist_train_and_val.shape[0]):
    label = labels_mnist_train_and_val[idx]
    if use_mnist:
        mnist_train_and_val[idx][-1] = label
        # bin_label = torch.tensor([int(d) for d in bin(label)[2:].zfill(4)]).float()
        # mnist_train_and_val[idx][-4:] = bin_label
    else:
        binary_label = 255 * torch.tensor([int(d) for d in bin(label)[2:].zfill(4)]).float()
        mnist_train_and_val[idx][-4:] = binary_label

# define train and validation
if use_mnist:
    X_train = mnist_train_and_val[0:50_000]
    X_val   = mnist_train_and_val[50_000::]
else: # if use_mnist is False then binarise
    X_train = (mnist_train_and_val[0:50_000] / 255 >= 0.5).float()
    X_val   = (mnist_train_and_val[50_000::] / 255 >= 0.5).float()

y_train = labels_mnist_train_and_val[0:50_000]
y_val   = labels_mnist_train_and_val[50_000::]

# load data into data loaders
batch_size = 128
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(X_val  , batch_size=batch_size)

# hyperparameters
# lamda = 0.0
latent_dim = 16
max_epochs = 100

# n_bins_list = [2 ** 14] # every element of this list corresponds to a model to train
n_chunks = 32 # if you run OOM, use n_chunks (e.g. n_chunks = 32)
seed_everything(42)
n_bins = 2 ** 13
# for latent_dim in [2, 4, 8, 16, 32]:
    # for lamda in [0, 0.2, 0.4, 0.6, 0.8, 1]:
for latent_dim in [32]:
    for lamda in [0.6, 0.8, 1]:

        # choose decoder architecture
        if use_mnist:
            decoder = CategoricalDecoder(
            net=mnist_conv_decoder(
                latent_dim=latent_dim,
                n_filters=64,
                out_channels=len(X_train.unique()),
                batch_norm=True)
            )
        else:
            decoder = BernoulliDecoder(
                net=get_decoder_debd(
                    latent_dim=latent_dim,
                    out_features=X_train.shape[1], # 28 * 28 = 784
                    n_layers=6,
                    batch_norm=True)
            )
        # decoder = CategoricalDecoder(
        #     net=mnist_conv_decoder(
        #         latent_dim=latent_dim,
        #         n_filters=64,
        #         out_channels=len(X_train.unique()),
        #         batch_norm=True)
        # )

        # model definition
        model = ContinuousMixture(
            decoder=decoder,
            sampler=GaussianQMCSampler(latent_dim, n_bins),
            lamda = lamda,
            X_num = 780,
            Y_num = 4,
        )
        model.n_chunks = n_chunks
        model.missing = False
        
        # early stopping set up
        cp_best_model_valid = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor='valid_loss_epoch',
            mode='min',
            filename='best_model_valid-{epoch}'
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="valid_loss_epoch",
            min_delta=0.00,
            patience=15,
            verbose=False,
            mode='min'
        )
        callbacks = [cp_best_model_valid, early_stop_callback]
        
        # trainer set up
        model_path = f'/logs/{dset}/hybrid/latent_dim_{latent_dim}/num_bins_{n_bins}/lambda_{lamda:.2f}'
        logger = pl.loggers.CSVLogger(
            save_dir=repo_dir+model_path,
            name=None,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if gpus and gpus > 0 else 'cpu',
            devices=gpus if gpus else 1,
            callbacks=callbacks,
            logger=logger,
            deterministic=True
        )

        # train model
        trainer.fit(model, train_loader, valid_loader)
        
        # get training and validation data
        log_version = len(next(os.walk(repo_dir+model_path))[1]) - 1
        df = pd.read_csv(repo_dir+model_path+f'/version_{log_version}/metrics.csv')
        loss_df = df.groupby("epoch")[["train_loss", "valid_loss_epoch"]].mean().reset_index()

        # plot training and validation losses
        plt.figure()
        plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Training Loss")
        plt.plot(loss_df["epoch"], loss_df["valid_loss_epoch"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title(f"Training and Validation Losses, latent_dim={latent_dim}")
        plt.tight_layout()

        # create figures directory and save training and validation plot inside
        os.makedirs(repo_dir+model_path+f"/version_{log_version}/figures")
        plt.savefig(repo_dir+model_path+f"/version_{log_version}/figures/train_val_loss_curves.png")
        plt.savefig(repo_dir+model_path+f"/version_{log_version}/figures/train_val_loss_curves.pdf")
