##############################################################################
#
# Copyright 2020-2022 Justus Sagemüller
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

import numpy as np
import torch
from ablation import compute_square_intensity, generate_grads
from ablation_paths import resample_to_reso

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


def pixel_ablation_sequence( model, x, baseline, saliency_method=None
                     , precomputed_saliency=None
                     , saliency_smoothing = lambda x: x
                     , ablmask_resolution = None
                     , nb_steps=8, path_steps=24, label_nr=None, submethod=None ):
    avg_grad = torch.mean(torch.cat([g.unsqueeze(0)
                    for g in list(generate_grads( model, x, baseline
                                                , nb_steps=nb_steps, label_nr=label_nr )
                                 )]), dim=0).reshape(x.shape)

    if saliency_method is None:
        if precomputed_saliency is None:
            raise ValueError("No saliency mask or standard method specified.")
        else:
            saliency = precomputed_saliency
    elif saliency_method in ['IG', 'IG-reverse']:
        saliency = ( compute_square_intensity(avg_grad)
                    * compute_square_intensity(x-baseline) )
    elif saliency_method in ["IG'", "IG'-reverse"]:
        saliency = torch.sum(avg_grad * (x-baseline), 0)
    elif saliency_method in ["IG''", "IG''-reverse"]:
        saliency = torch.abs(torch.sum(avg_grad * (x-baseline), 0))
    elif saliency_method in ['AG', 'AG-reverse', 'random']:
        saliency = compute_square_intensity(avg_grad)
    else:
        raise ValueError("Unknown saliency method '%s'" % saliency_method)

    if saliency_method == 'random':
        saliency = torch.rand_like(saliency)
    
    imgW, imgH = saliency.shape
    saliency = saliency_smoothing(saliency.reshape([1,imgW,imgH])).reshape([imgW,imgH])
    
    ranking = multi_argsort(saliency.cpu().detach().numpy())

    if saliency_method in ['IG-reverse', "IG'-reverse", 'AG-reverse']:
        ranking = np.flip(ranking,0)
    mask = torch.zeros_like(saliency)

    n_pixels = imgW * imgH

    if ablmask_resolution is None:
        ablmask_resolution = (imgW, imgH)

    next_step = 1
    for ia, pos in enumerate(ranking):
        mask[pos[0],pos[1]] = 1
        if ia > next_step*n_pixels/(path_steps+1):
            next_step += 1
            timeslice = resample_to_reso(mask.clone(), ablmask_resolution).clone().detach()
            assert timeslice.shape == ablmask_resolution, (timeslice.shape, mask.shape, x.shape)
            yield timeslice
