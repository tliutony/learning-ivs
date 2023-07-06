import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing

import numpy as np
import pandas as pd
import seaborn as sns

from ipywidgets import interact_manual, IntSlider, FloatSlider
from sklearn.linear_model import LinearRegression
from linearmodels.iv.model import IV2SLS
from linearmodels.iv.model import _OLS

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def generate_const_linear_iv(
    n_samples,
    seed,
    pi,
    psi,
    tau,
    gamma
):
    """
    Generates linear IV with constant treatment effects.
    
    Args:
        n_samples (int): num samples to generate
        seed (int): seed for reproducibilty
        pi (float): instrument "strength"
        psi (float): confounding "strength"
        tau (float): treatment effect
        gamma (float): confound effect
    
    Returns:
        pd.DataFrame
    """
    np.random.seed(seed),
    Z = np.random.normal(0, 1, size=n_samples)#np.random.uniform(0, 10, n_samples)
    C = np.random.normal(0, 1, size=n_samples)#np.random.uniform(0, 10, n_samples)
    eta = np.random.normal(0, 1, size=n_samples)
    const = np.random.uniform(-1, 1)

    T = const + (pi*Z) + (psi*C) + eta

    epsilon = np.random.normal(0, 1, size=n_samples)
    beta = np.random.uniform(-1, 1)

    Y = beta + (tau*T) + (gamma*C) + epsilon

    data = np.concatenate([Z.reshape(-1,1), 
                           C.reshape(-1,1), 
                           T.reshape(-1,1),
                           Y.reshape(-1,1),], 
                         axis=1)

    data_df = pd.DataFrame(data, columns=['Z', 'C', 'T', 'Y'])

    return data_df

def gen_datasets():
    datasets = {

    }

    n_datasets = 10000
    n_test_datasets = 2000
    iv_strs = np.round(np.linspace(0, 2, 11), 2) 

    for pi in tqdm(iv_strs):
        datasets[pi] = {
            "data": [],
            "taus": np.zeros(n_datasets),
            "confounds": np.zeros(n_datasets),
            "data_tup": [],
        }
        # data = []#np.zeros((n_datasets, 10))
        # taus = np.zeros(n_datasets)
        # confounds = np.zeros(n_datasets)
        
        for i in range(n_datasets):
            seed = i + int(pi*10000) # to ensure we have non-overlapping datasets
            np.random.seed(seed)
            treat_effect = np.random.uniform(-2, 2)
            confound_effect = 5 #np.random.uniform(1, 5) # hold confounding constant, for now TODO
            psi_effect = np.random.uniform(5, 10)
    
            n_samples = 1000
            
            data_df = generate_const_linear_iv(
                n_samples=n_samples,
                seed=seed,
                pi=pi,
                psi=psi_effect,
                tau=treat_effect,
                gamma=confound_effect)
            
            # feats = generate_iv_features(data_df)
            # data[i,:] = feats

            # # zero out the variance of Y and variance of T
            # data[i, feat_cols.index("var_Y")] = 0
            # data[i, feat_cols.index("var_T")] = 0

            datasets[pi]['data'].append(data_df)
            datasets[pi]['taus'][i] = treat_effect
            datasets[pi]['confounds'][i] = confound_effect

            # convert data_df and tau to torch dataloader
            datasets[pi]['data_tup'].append((data_df.drop("C", axis='columns').values.astype('float32'), treat_effect))

            # convert datasets data and taus to torch dataloader
            train_data = torch.utils.data.DataLoader(
                datasets[pi]['data_tup'][:n_datasets - n_test_datasets],
                batch_size=32,
            )

            test_data = torch.utils.data.DataLoader(
                datasets[pi]['data_tup'][n_test_datasets:],
                batch_size=n_test_datasets,
            )

            datasets[pi]['train_data'] = train_data
            datasets[pi]['test_data'] = train_data

    return datasets


# Joint autoencoder + treatment effect model
class JointAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(JointAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.treatment = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(-1, self.input_dim)
        z = self.encoder(x)
        tau_hat = self.treatment(z)
        x_hat = self.decoder(z)
        return x_hat, z, tau_hat
    

# train the joint autoencoder
def train_joint_autoencoder(
    model,
    train_data,
    test_data,
    num_epochs,
    lr,
    device,
    verbose=False,
    batch_size=32,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    test_x, test_y = next(iter(test_data))
    test_x = test_x.to(torch.float32).to(device)
    test_y = test_y.to(torch.float32).to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, labels) in enumerate(train_data):
            data = data.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)

            optimizer.zero_grad()

            x_hat, z, tau_hat = model(data)
            x_hat = x_hat.view(batch_size, -1, 3)
            loss = criterion(x_hat, data) + criterion(tau_hat.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        model.eval()
        with torch.no_grad():
            _, _, tau_hat = model(test_x)
            tau_hat = tau_hat.squeeze()
            test_loss = criterion(tau_hat, test_y)

        if verbose and epoch %  10 == 0:
            print(f"Epoch {epoch} loss: {running_loss}, test MSE loss: {test_loss}")
            
    return model

if __name__ == "__main__":
    # generate datasets
    n_samples = 1000
    datasets = gen_datasets()

    # initialize the joint autoencoder
    input_dim = 3 * n_samples
    latent_dim = 20
    hidden_dim = 128
    num_epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for pi in datasets.keys():
        print(pi)
        train_data = datasets[pi]['train_data']
        test_data = datasets[pi]['test_data']
        model = JointAutoencoder(input_dim, latent_dim, hidden_dim)
        model = train_joint_autoencoder(
            model,
            train_data,
            test_data,
            num_epochs,
            lr,
            device,
            verbose=True,
        )
        torch.save(model, f"joint_autoencoder_{pi}.pt")