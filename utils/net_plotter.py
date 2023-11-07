import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from matplotlib import cm
import matplotlib.pyplot as plt

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size(), device=w.device) for w in weights]


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in states.items()]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]

def get_normalized_weight_rand_direction(weights):
    random_direction1 = get_random_weights(weights)
    random_direction2 = get_random_weights(weights)

    for d1,d2,w in zip(random_direction1,random_direction2,weights):
        
        w_norm  = w.view((w.shape[0],-1)).norm(dim=(1),keepdim=True).squeeze(1)
        d_norm1 = d1.view((d1.shape[0],-1)).norm(dim=(1),keepdim=True).squeeze(1)
        d_norm2 = d2.view((d2.shape[0],-1)).norm(dim=(1),keepdim=True).squeeze(1)

        scale_factor_1 = (w_norm/(d_norm1+1e-10))
        scale_factor_2 = (w_norm/(d_norm2+1e-10))
        for i in range(d1.dim() - 1):
            scale_factor_1 = scale_factor_1.unsqueeze(-1)
            scale_factor_2 = scale_factor_2.unsqueeze(-1)
        d1.data = d1 * scale_factor_1
        d2.data = d2 * scale_factor_2
    return random_direction1, random_direction2

def get_normalized_state_rand_direction(states):
    random_direction1 = get_random_states(states)
    random_direction2 = get_random_states(states)

    for d1,d2,w in zip(random_direction1,random_direction2,states):
        
        w_norm  = w.view((w.shape[0],-1)).norm(dim=(1),keepdim=True)
        d_norm1 = d1.view((d1.shape[0],-1)).norm(dim=(1),keepdim=True)
        d_norm2 = d2.view((d2.shape[0],-1)).norm(dim=(1),keepdim=True)

        d1.data = d1 * (w_norm/(d_norm1+1e-10))
        d2.data = d2 * (w_norm/(d_norm2+1e-10))
    return random_direction1, random_direction2

def plot_3D(x_l, x_h, y_l, y_h, model, test_dataloader, criterion, device):
    x = np.arange(x_l, x_h, step=(x_h - x_l) / 10)
    y = np.arange(y_l, y_h, step=(y_h - y_l) / 10)
    #alpha, beta = torch.meshgrid(torch.tensor(x),torch.tensor(y))
    model.eval()
    weight = get_weights(model)
    dir_1, dir_2 = get_normalized_weight_rand_direction(weight)
    dir_1 = parameters_to_vector(dir_1)
    dir_2 = parameters_to_vector(dir_2)
    weight = parameters_to_vector(weight)
    
    def tau_2d(alpha, beta, theta_ast):
        space = alpha * dir_1 + beta * dir_2 + theta_ast
        return space

    losses = torch.zeros(len(x), len(y))
    print(f"x : {x}")
    print(f"y : {y}")
    for a, _ in enumerate(x):
        print(f'a = {a}', end=" ")
        for b, _ in enumerate(y):
            print(f'b = {b}', end=" ")
            print(f'alpha = {x[a]}, beta = {y[b]}')
            vector_to_parameters(tau_2d(x[a], y[b], weight), model.parameters())
            for iterations, (image, label) in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    image = image.to(device)
                    label = label.to(device)
                    losses[a][b] += criterion(model(image), label).item()
            losses[a][b] /= len(test_dataloader)
            print(f"Loss at ({a}, {b}) is {losses[a][b]}")
        
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, losses.numpy(), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.savefig("loss_cur.jpg")