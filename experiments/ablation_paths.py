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
import odl
from monotone_paths import project_monotone_lInftymin, IntegrationOperator
from ablation import compute_square_intensity
from image_filtering import apply_filter
from imagenet_loading import load_single_image
from itertools import count

def all_indices(t):
    result = list([(k,) for k in range(t.shape[0])])
    for i in range(1, len(t.shape)):
        result_aug = [ixs+(k,) for ixs in result for k in range(t.shape[i])]
        result = list(result_aug)
    return result

def monotonise_ablationpath(abl_seq):
    if type(abl_seq) is torch.Tensor:
        # This should be generalised to work with any tensor shape,
        # not just two spatial dimensions
        for i,j in all_indices(abl_seq[0]):
            thispixel = abl_seq[:,i,j].cpu().numpy()
            project_monotone_lInftymin(thispixel)
            abl_seq[:,i,j] = torch.tensor(thispixel)
    elif type(abl_seq) is np.ndarray:
        for i,j in all_indices(abl_seq[0]):
            thispixel = abl_seq[:,i,j]
            project_monotone_lInftymin(thispixel)
            abl_seq[:,i,j] = thispixel
    elif type(abl_seq) is odl.DiscretizedSpaceElement:
        ablseq_arr = abl_seq.asarray()
        for ij in all_indices(ablseq_arr[0]):
            pixslice = (np.s_[:],) + ij
            thispixel = ablseq_arr[pixslice]
            project_monotone_lInftymin(thispixel)
            ablseq_arr[pixslice] = thispixel
        abl_seq = abl_seq.space.element(ablseq_arr)
    else:
        raise ValueError("This function currently works only with Torch tensors, Numpy arrays or ODL DiscretizedSpace elements.")

def reParamNormalise_ablation_speed(abl_seq):
    n = abl_seq.shape[0]
    masses = np.array([0] + [float(abl_seq[i].mean()) for i in range(n)] + [1])
    zeros_like = (torch.zeros_like
           if type(abl_seq) is torch.Tensor
            else np.zeros_like )
    ones_like = (torch.ones_like
           if type(abl_seq) is torch.Tensor
            else np.ones_like )
    result = zeros_like(abl_seq)
    il = 0
    ir = 1
    for j, m in enumerate(np.linspace(0, 1, n+2)[1:-1]):
        while ir<n+1 and masses[ir]<=m:
            ir+=1
        while il<ir-1 and masses[il+1]<m:
            il+=1
        η = (m - masses[il]) / (masses[ir]-masses[il])
        # print("m=%.2f, il=%i, mil=%.2f, ir=%i, mir=%.2f" % (m, il, masses[il], ir, masses[ir]))
        φl = abl_seq[il-1] if il>0 else zeros_like(abl_seq[0])
        φr = abl_seq[ir-1] if ir<=n else ones_like(abl_seq[0])
        result[j] = φl + (φr-φl)*η
    return result

# Given a possibly invalid path of ablation-masks (i.e., one that may not be
# pointwise monotone, in the allowed range [0,1], or speed-normalised),
# return a path that is similar but does fulfill the conditions.
# Note that the argument of this function is mutated.
def repair_ablation_path(abl_seq):
    monotonise_ablationpath(abl_seq)
    if type(abl_seq) is torch.Tensor:
        torch.clamp(abl_seq, 0, 1, out=abl_seq)
        return reParamNormalise_ablation_speed(abl_seq)
    elif type(abl_seq) is np.ndarray:
        return reParamNormalise_ablation_speed(np.clip(abl_seq, 0, 1))
    elif type(abl_seq) is odl.DiscretizedSpaceElement:
        abl_arr = abl_seq.asarray()
        return abl_seq.space.element(
            reParamNormalise_ablation_speed(np.clip(abl_arr, 0, 1)))


def time_pderiv(dom):
    return odl.PartialDerivative(dom, axis=0, pad_mode='order1')
def time_cumu_integral(dom):
    intg_cell_vol = dom.partition.byaxis[0].cell_volume
    def integrate(ψ):
        ψData = np.roll(ψ.asarray(), 1, axis=0)
        ψData[0] = ψData[1]/2   # trapezoidal and corresponding to
                                # order-0 extension of PartialDerivative.
        return dom.element(np.cumsum(ψData, axis=0) * intg_cell_vol )
    return integrate

def unitIntegralConstraint(intg_op):
    integrationField = intg_op.range
    return odl.solvers.functional.IndicatorZero(integrationField
                        ).translated(integrationField.element(lambda x: x[0]*0 + 1))

def dist2(ψ):
    return odl.solvers.functional.L2Norm(ψ.space).translated(ψ)

def repair_ablation_path_convexOpt( φ, distancespace_embedding=None
                                  , extra_penalty_ops=[], iterations=20
                                  ):
    usesODL = type(φ) is odl.DiscretizedSpaceElement
    usesTorch = type(φ) is torch.Tensor
    torchdevice = φ.device if usesTorch else None
    space = φ.space if usesODL else (
        odl.uniform_discr(min_pt=[0 for _ in φ.shape], max_pt=[1 for _ in φ.shape]
                   , shape=φ.shape, dtype='float32') )
    if usesTorch:
        φ = φ.cpu().numpy()
    if not usesODL:
        φ = space.element(φ)
    if distancespace_embedding is None:
        distancespace_embedding = odl.IdentityOperator(space)
    elif distancespace_embedding=='φspace-L²':
        distancespace_embedding = IntegrationOperator(space, cumu_intg_directions=(0,))
    nonnegativity = odl.solvers.functional.IndicatorNonnegativity(space)
    integration_time = IntegrationOperator(space, integration_directions=(0,))
    unitIntegral_time = unitIntegralConstraint(integration_time)
    integration_space = IntegrationOperator(
          space, integration_directions=tuple(range(1, len(space.shape))) )
    unitIntegral_space = unitIntegralConstraint(integration_space)
    ψOrig = time_pderiv(space)(φ)
    ψTgt = distancespace_embedding(ψOrig)
    ψ = ψOrig.copy()
    odl.solvers.nonsmooth.pdhg(
        ψ
      , nonnegativity
      , odl.solvers.functional.SeparableSum(
                               dist2(ψTgt)
                             , unitIntegral_space, unitIntegral_time
                             , *[ op[1] if isinstance(op, tuple)
                                   else odl.solvers.functional.IdentityFunctional(op.range)
                                  for op in extra_penalty_ops]
                             )
      , odl.BroadcastOperator( distancespace_embedding
                             , integration_space, integration_time
                             , *[ op[0] if isinstance(op, tuple)
                                   else 0
                                  for op in extra_penalty_ops] )
      , iterations )
    result = time_cumu_integral(φ.space)(ψ)
    if usesTorch:
        return torch.tensor(result.asarray()).to(torchdevice)
    elif usesODL:
        return result
    else:
        return result.asarray()

def resample_to_reso(v, tgt_shape):
    if len(v.shape) < 4:
        return resample_to_reso(v.unsqueeze(1), tgt_shape).squeeze(1)
    elif v.shape[2:] == tgt_shape:
        return v
    else:
        assert(len(tgt_shape)==2)
        return torch.nn.functional.interpolate( v, size=tgt_shape
                                              , mode='bilinear', align_corners=False )
        


def gradientMove_ablation_path( model, x, baseline, abl_seq, optstep, label_nr=None
                              , pointwise_scalar_product=False, gradients_postproc=lambda gs: gs
                              ):
    needs_resampling = x.shape[1:] != abl_seq.shape[1:]

    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))
    elif label_nr=='baseline_label':
        label_nr = torch.argmax(model(baseline.unsqueeze(0)))
    nSq, wMask, hMask = abl_seq.shape
    nCh, wX, hX = x.shape

    ch_rpl_seq = resample_to_reso(abl_seq.reshape(nSq,1,wMask,hMask), (wX, hX)
                      ).repeat(1,nCh,1,1)
    xOpt = x.to(abl_seq.device)
    difference = baseline.to(abl_seq.device) - xOpt
    
    # The path score, which is to be computed as an integral.
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
        abl_seq += resample_to_reso(abl_update, (wMask,hMask))
    else:
        abl_seq += abl_update
    return intg

def saturated_masks(φ, saturation):
    return (torch.tanh( (φ*2 - torch.ones_like(φ))*saturation )
                          / (np.tanh(saturation))
                       + torch.ones_like(φ))/2

def path_optimisation_sequence (
          model, x, baselines, path_steps, optstep
        , saturation=0, filter_cfg=0, filter_mix_ratio=1
        , initpth=None, ablmask_resolution=None
        , pathrepairer=repair_ablation_path
        , momentum_inertia=0
        , **kwargs):
    if ablmask_resolution is None:
        ablmask_resolution = x.shape[1:]
    pth = ( torch.stack([p*torch.ones(ablmask_resolution)
                         for p in np.linspace(0,1,path_steps)[1:-1]])
                   .to(x.device)
             if initpth is None else initpth )
    if momentum_inertia>0:
        momentum = torch.zeros_like(pth)

    for i in count():
        if momentum_inertia>0:
            old_pth = pth.clone().detach()
        current_score = gradientMove_ablation_path(
            model, x, baselines(), abl_seq=pth, optstep=optstep, **kwargs )
        if momentum_inertia>0:
            momentum = ( momentum * momentum_inertia
                        + (pth - old_pth)*(1-momentum_inertia) )
            pth = old_pth + momentum
        if saturation>0:
            pth = saturate_masks(pth,saturation)
        def filterWith(σ):
            if ablmask_resolution is not None:
                wMask, hMask = ablmask_resolution
                w, h = x.shape[1:]
                scale_factor = np.sqrt(wMask*hMask/(w*h))
            nonlocal pth
            pth = pth*(1-filter_mix_ratio) + apply_filter(pth, σ*scale_factor)*filter_mix_ratio
        if callable(filter_cfg):
            filterWith(filter_cfg(i))
        elif filter_cfg>0:
            filterWith(filter_cfg)
        pth = pathrepairer(pth)
        if momentum_inertia>0:
            momentum = pth - old_pth
        yield pth, current_score

def optimised_path( model, x, baselines, path_steps, optstep
                  , iterations, abort_criterion=(lambda scr: False)
                  , **kwargs):
    i = 0
    for pth, current_score in path_optimisation_sequence (
          model, x, baselines, path_steps, optstep, **kwargs ):
        if i>=iterations or abort_criterion(current_score):
            return pth
        print(current_score)
        i+=1

def masked_interpolation(x, baseline, abl_seq, include_endpoints=False):
    needs_resampling = x.shape[1:] != abl_seq.shape[1:]
    if type(abl_seq) != torch.Tensor:
        abl_seq = torch.stack(list(abl_seq))
    xOpt = x.to(abl_seq.device)
    nSq, wMask, hMask = abl_seq.shape
    nCh, wX, hX = x.shape

    ch_rpl_seq = resample_to_reso(abl_seq.reshape(nSq,1,wMask,hMask), (wX, hX)
                      ).repeat(1,nCh,1,1)

    difference = baseline.to(abl_seq.device) - xOpt
    if include_endpoints:
        return ( [x] + [ (xOpt + difference.to(abl_seq.device)*ch_rpl_seq[i]
                   ).detach()
                  for i in range(nSq) ]
                 + [baseline] )
    else:
        return [ (xOpt + difference.to(abl_seq.device)*ch_rpl_seq[i]
                   ).detach()
                  for i in range(nSq) ]

def find_class_transition( model, x, baseline, abl_seq
                         , minimum_ablation_pos=0.25, label_nr=None ):
    if len(abl_seq) <= 1:
        return 0

    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))

    predictions = model(torch.stack(
                               masked_interpolation(x,baseline,abl_seq)))
    imax = len(abl_seq) - 1
    min_allowed_i = int(len(abl_seq) * minimum_ablation_pos)
    while imax>=min_allowed_i and torch.argmax(predictions[imax])!=label_nr:
        imax -= 1
    if imax < min_allowed_i:
        best_score = 0
        imax = min_allowed_i
        for i in range(min_allowed_i, len(abl_seq)):
            score = torch.softmax(predictions[i], 0)[label_nr]
            if score > best_score:
                score = best_score
                imax = i
    print("imax = %i" % imax)
    return imax

def load_ablation_path_from_images(fns, size_spec=None, path_steps=None, torchdevice=None):
    if size_spec is None:
        reference, _ = load_single_image(fns[0], greyscale=True)
        size_spec = tuple(reference.shape)
    path_masks = [ load_single_image(fn, size_spec=size_spec, greyscale=True)[0]
                      for fn in fns ]
    path_masks.sort(key = lambda mask: mask.mean())
    if path_steps is not None and path_steps > len(path_masks):
        path_masks = [ path_masks[0] for _ in range(path_steps - len(path_masks))
                     ] + path_masks
    path_candidate = torch.stack(path_masks)
    if torchdevice is not None:
        path_candidate = path_candidate.to(torchdevice)
    path_candidate = repair_ablation_path(path_candidate)
    if path_steps is not None and path_steps < len(path_masks):
        path_candidate = torch.stack(
           [ path_candidate[int(len(path_masks)*j/path_steps)]
            for j in range(path_steps) ])
    return path_candidate
