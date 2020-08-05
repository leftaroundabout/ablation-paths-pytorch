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

def multi_argsort(arr):
    lin_indices = arr.flatten().argsort()
    def produce_ixs():
        nonlocal lin_indices
        quotr = len(lin_indices)
        for dlen in arr.shape:
            quotr = quotr // dlen
            yield (lin_indices // quotr)
            lin_indices = lin_indices % quotr
    return np.stack(list(produce_ixs())).transpose()

def ablation_sequence( model, x, baseline, saliency_method, ablation_target_gen
                     , nb_steps=8, ablPixs_per_step=800, label_nr=None ):
    avg_grad = torch.mean(torch.cat([g.unsqueeze(0)
                    for g in list(generate_grads( model, x, baseline
                                                , nb_steps=nb_steps, label_nr=label_nr )
                                 )]), dim=0).reshape(x.shape)
    saliency = compute_square_intensity(avg_grad)

    if saliency_method in ['IG', 'IG-reverse']:
        saliency = saliency * compute_square_intensity(x-baseline)
    elif saliency_method in ['AG', 'AG-reverse']:
        pass
    elif saliency_method is 'random':
        saliency = torch.rand_like(saliency)
    else:
        print("Unknown saliency method '%s'" % saliency_method)
    
    ablation_target = ablation_target_gen(x)

    ranking = multi_argsort(saliency.cpu().detach().numpy())

    if saliency_method in ['IG-reverse', 'AG-reverse']:
        ranking = np.flip(ranking,0)
    ablatee = x.clone()
    for ia, pos in enumerate(ranking):
        ablatee[:,pos[0],pos[1]] = ablation_target[:,pos[0],pos[1]]
        if ia % ablPixs_per_step is 0:
            yield ablatee.clone()
