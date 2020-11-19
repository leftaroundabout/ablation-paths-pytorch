##############################################################################
#
# Copyright 2020 Justus Sagemüller
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
from monotone_paths import project_monotone_lInftymin
from ablation import compute_square_intensity

def all_indices(t):
    result = list([(k,) for k in range(t.shape[0])])
    for i in range(1, len(t.shape)):
        result_aug = [ixs+(k,) for ixs in result for k in range(t.shape[i])]
        result = list(result_aug)
    return result

def monotonise_ablationpath(abl_seq):
    # This should be generalised to work with any tensor shape,
    # not just two spatial dimensions
    for i,j in all_indices(abl_seq[0]):
        thispixel = abl_seq[:,i,j].cpu().numpy()
        project_monotone_lInftymin(thispixel)
        abl_seq[:,i,j] = torch.tensor(thispixel)

def reParamNormalise_ablation_speed(abl_seq):
    n = abl_seq.shape[0]
    masses = np.array([0] + [float(abl_seq[i].mean()) for i in range(n)] + [1])
    result = torch.zeros_like(abl_seq)
    il = 0
    ir = 1
    for j, m in enumerate(np.linspace(0, 1, n+2)[1:-1]):
        while ir<n+1 and masses[ir]<=m:
            ir+=1
        while il<ir-1 and masses[il+1]<m:
            il+=1
        η = (m - masses[il]) / (masses[ir]-masses[il])
        # print("m=%.2f, il=%i, mil=%.2f, ir=%i, mir=%.2f" % (m, il, masses[il], ir, masses[ir]))
        φl = abl_seq[il-1] if il>0 else torch.zeros_like(abl_seq[0])
        φr = abl_seq[ir-1] if ir<=n else torch.ones_like(abl_seq[0])
        result[j] = φl + (φr-φl)*η
    return result

# Given a possibly invalid path of ablation-masks (i.e., one that may not be
# pointwise monotone, in the allowed range [0,1], or speed-normalised),
# return a path that is similar but does fulfill the conditions.
# Note that the argument of this function is mutated.
def repair_ablation_path(abl_seq):
    monotonise_ablationpath(abl_seq)
    torch.clamp(abl_seq, 0, 1, out=abl_seq)
    return reParamNormalise_ablation_speed(abl_seq)


def gradientMove_ablation_path( model, x, baseline, abl_seq, optstep, label_nr=None
                              , pointwise_scalar_product=False, gradients_postproc=lambda gs: gs):
    needs_resampling = x.shape[1:] != abl_seq.shape[1:]

    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))
    nSq, wMask, hMask = abl_seq.shape
    nCh, wX, hX = x.shape

    ch_rpl_seq = abl_seq.reshape(nSq,1,wMask,hMask)
    if needs_resampling:
        ch_rpl_seq = torch.nn.functional.interpolate(ch_rpl_seq, size=x.shape[1:]
                                                     , mode='bilinear', align_corners=False)
    ch_rpl_seq = ch_rpl_seq.repeat(1,nCh,1,1)
    xOpt = x.to(abl_seq.device)
    difference = baseline.to(abl_seq.device) - xOpt
    intg = 0
    gs = torch.zeros(nSq, nCh, wX, hX).to(abl_seq.device)
    for i in range(nSq):
        argument = (xOpt + difference.to(abl_seq.device)*ch_rpl_seq[i]
                   ).detach().unsqueeze(0)
        argument.requires_grad = True
        result = torch.softmax(model(argument)[0], 0)
        gs[i] = torch.autograd.grad(result[label_nr], argument)[0].squeeze(0)
        intg += float(result[label_nr])/abl_seq.shape[0]

    gs = gradients_postproc(gs)

    abl_update = torch.zeros(nSq, wX, hX).to(abl_seq.device)
    for i in range(nSq):
        direction = ( torch.sum(gs[i] * difference, 0)
                       if pointwise_scalar_product
                       else -torch.sqrt(compute_square_intensity(gs[i])) )
        direction -= torch.sum(direction)/torch.sum(torch.ones_like(direction))
        if optstep is None:
            abl_update[i] = direction/(torch.max(direction) - torch.min(direction))
        else:
            abl_update[i] = optstep*direction
    if needs_resampling:
        abl_seq += torch.nn.functional.interpolate(abl_update.unsqueeze(1)
                                                   , size=(wMask,hMask)
                                                   , mode='bilinear', align_corners=False
                                                   ).squeeze(1)
    else:
        abl_seq += abl_update
    return intg


def optimised_path( model, x, baselines, path_steps, optstep, iterations
                  , saturation=0, filter_sigma=0, filter_eta=1
                  , initpth=None, ablmask_resolution=None
                  , **kwargs):
    if ablmask_resolution is None:
        ablmask_resolution = x.shape[1:]
    pth = ( torch.stack([p*torch.ones(ablmask_resolution)
                         for p in np.linspace(0,1,path_steps)[1:-1]])
                   .to(x.device)
             if initpth is None else initpth )

    for i in range(iterations):
        print(gradientMove_ablation_path( model, x, baselines(), abl_seq=pth, optstep=optstep, **kwargs ))
        if saturation>0:
            pth = (torch.tanh( (pth*2 - torch.ones_like(pth))*saturation )
                          / (np.tanh(saturation))
                    + torch.ones_like(pth))/2
        def filterWith(σ):
            nonlocal pth
            pth = pth*(1-filter_eta) + apply_filter(pth, σ)*filter_eta
        if type(filter_sigma) is type(lambda i: 0):
            filterWith(filter_sigma(i))
        elif filter_sigma>0:
            filterWith(filter_sigma)
        pth = repair_ablation_path(pth)
    return pth

def masked_interpolation(x, baseline, abl_seq):
    if type(abl_seq) != torch.Tensor:
        abl_seq = torch.stack(list(abl_seq))
    xOpt = x.to(abl_seq.device)
    nSq, w, h = abl_seq.shape
    nCh = x.shape[0]

    ch_rpl_seq = abl_seq.reshape(nSq,1,w,h).repeat(1,nCh,1,1)

    difference = baseline.to(abl_seq.device) - xOpt
    return [ (xOpt + difference.to(abl_seq.device)*ch_rpl_seq[i]
                   ).detach()
                  for i in range(nSq) ]

def find_class_transition(model, x, baseline, abl_seq, label_nr=None):
    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))

    predictions = model(torch.stack(
                               masked_interpolation(x,baseline,abl_seq)))
    imax = len(abl_seq) - 1
    while torch.argmax(predictions[imax])!=label_nr:
        imax -= 1
    return imax


