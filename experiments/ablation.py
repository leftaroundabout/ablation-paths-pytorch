import torch
import numpy as np
import itertools

def interpolate(im0, im1, t):
    return im0*(1-t) + im1*t

def generate_grads(model, x, baseline, nb_steps=4, label_nr=None):
    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))
    steps = np.linspace(0,1,nb_steps)
    for s in steps:
        interp = interpolate(baseline, x, s).unsqueeze(0)
        interp.requires_grad = True
        g = torch.autograd.grad(model(interp)[0,label_nr], interp)
        yield torch.squeeze(g[0])

def compute_square_intensity(tensor):
    return torch.sum(tensor * tensor, dim=0)

