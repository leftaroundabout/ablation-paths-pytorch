##############################################################################
#
# Copyright 2020-2023 Justus Sagemüller
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
from monotone_paths import project_monotone_lInftymin, IntegrationOperator
from ablation import compute_square_intensity
from image_filtering import apply_filter, FilteringConfig, LowpassFilterType, NOPFilteringConfig
from imagenet_loading import load_single_image
from itertools import count
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from numbers import Number

def all_indices(t):
    result = list([(k,) for k in range(t.shape[0])])
    for i in range(1, len(t.shape)):
        result_aug = [ixs+(k,) for ixs in result for k in range(t.shape[i])]
        result = list(result_aug)
    return result

def monotonise_ablationpath(abl_seq):
    if type(abl_seq) is torch.Tensor:
        assert(len(abl_seq.shape)>1), f"{abl_seq.shape=}"
        for ij in all_indices(abl_seq[0]):
            slicesel = (slice(None), *ij)
            thispixel = abl_seq[slicesel].cpu().numpy()
            project_monotone_lInftymin(thispixel)
            abl_seq[slicesel] = torch.tensor(thispixel)
    elif type(abl_seq) is np.ndarray:
        for ij in all_indices(abl_seq[0]):
            slicesel = (slice(None), *ij)
            thispixel = abl_seq[slicesel]
            project_monotone_lInftymin(thispixel)
            abl_seq[slicesel] = thispixel
    else:
        raise ValueError("This function currently works only with Torch tensors, Numpy arrays or ODL DiscretizedSpace elements.")

class PathsSpace(ABC):
    """An abstract notion of what it means to interpolate along paths of
    masks in order to obtain paths of images. Corresponds roughly to what
    [Fong&Vedaldi 2017] call “pertubations”.
    
    The simplest such notion – linear interpolation from the target image
    to a baseline – is captured by `LinInterpPathsSpace`."""
    @abstractmethod
    def apply_mask_seq(self, abl_seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @property
    @abstractmethod
    def target_image(self):
        raise NotImplementedError
    @property
    @abstractmethod
    def baseline_image(self):
        raise NotImplementedError
    @property
    @abstractmethod
    def ablmask_shape(self):
        raise NotImplementedError
    def differential_to_gradient(self, differential: torch.Tensor) -> torch.Tensor:
        """ Given the dual vector that comes out of e.g. Torch autograd,
        obtain the corresponding gradient as a vector that can be used as
        an update in gradient-descent. This can be seen as incurring a
        norm / inner product on the space of paths.
        The default simply uses the differential itself, as is valid for
        Euclidean spaces (which are self-dual). """
        return differential
    def mask_masses(self, abl_seq: torch.Tensor) -> torch.Tensor:
        n_mask_dims = len(self.ablmask_shape)
        return abl_seq.mean(dim=tuple(range(-n_mask_dims,0)))

class RangeRemapping(ABC):
    @abstractmethod
    def to_unitinterval(self, x):
        return NotImplemented
    @abstractmethod
    def from_unitinterval(self, y):
        return NotImplemented

def reParamNormalise_ablation_speed( abl_seq, pathspace: PathsSpace
                                   , range_remapping: RangeRemapping
                                   , accuracy: float =0.01
                                   , max_interpfind_iters: int =200
                                   , endpoint_limitε: float =1e-1 ):
    n = abl_seq.shape[0]
    uses_torch = isinstance(abl_seq, torch.Tensor)
    zeros = (torch.zeros if uses_torch else np.zeros )
    ablseq_wends = (torch.cat if uses_torch else np.concat
            )([ (abl_seq[0]*endpoint_limitε)[None]
              , abl_seq
              , (1 - (1 - abl_seq[-1])*endpoint_limitε)[None] ])
    masses = pathspace.mask_masses(ablseq_wends)
    assert(masses.shape[0]==n+2)
    batchdims = masses.shape[1:]
    assert(ablseq_wends.shape[1 : 1+len(batchdims)]==batchdims
          ), f"{ablseq_wends.shape=}, {batchdims=}"
    mask_shape = ablseq_wends.shape[1+len(batchdims):]
    # assert(mask_shape==pathspace.ablmask_shape
    #       ), f"{mask_shape=}, {pathspace.ablmask_shape=}"
    n_batches = np.product(batchdims) if len(batchdims)>0 else 1
    masses = masses.reshape(n+2, n_batches)
    ablseq_wends = ablseq_wends.reshape(n+2, n_batches, *mask_shape)
    result = zeros((n,n_batches)+mask_shape).to(abl_seq.device)
    for k in range(n_batches):
        il = 0
        ir = 1
        for j, m in enumerate(np.linspace(0, 1, n+2)[1:-1]):
            while ir<n+1 and masses[ir][k]<=m:
                ir+=1
            while il<ir-1 and masses[il+1][k]<m:
                il+=1
            assert(masses[il][k] < m and masses[ir][k] > m
                  ), f"{m=}, {masses[il][k]=}, {masses[ir][k]=}"
            η = (m - masses[il][k]) / (masses[ir][k]-masses[il][k])
            φl = ablseq_wends[il][k]
            φlω = range_remapping.from_unitinterval(φl)
            φr = ablseq_wends[ir][k]
            φrω = range_remapping.from_unitinterval(φr)
            def interpolation_point(τgl, τgr, iters=0):
                τgm = (τgl+τgr)/2
                φgmω = φlω*(1-τgm) + φrω*τgm
                φgm = range_remapping.to_unitinterval(φgmω)
                mgm = pathspace.mask_masses(φgm)
                if abs(mgm-m) < accuracy or iters>max_interpfind_iters:
                    return φgm
                elif m<mgm:
                    return interpolation_point(τgl, τgm, iters+1)
                else:
                    return interpolation_point(τgm, τgr, iters+1)
            result[j][k] = interpolation_point(0,1)
    return result.reshape((n,)+batchdims+mask_shape)

# Given a possibly invalid path of ablation-masks (i.e., one that may not be
# pointwise monotone, in the allowed range [0,1], or speed-normalised),
# return a path that is similar but does fulfill the conditions.
# Note that the argument of this function is mutated.
def repair_ablation_path(abl_seq, pathspace: PathsSpace, range_remapping: RangeRemapping):
    monotonise_ablationpath(abl_seq)
    if type(abl_seq) is torch.Tensor:
        torch.clamp(abl_seq, 0, 1, out=abl_seq)
        return reParamNormalise_ablation_speed(abl_seq, pathspace, range_remapping)
    elif type(abl_seq) is np.ndarray:
        return reParamNormalise_ablation_speed(np.clip(abl_seq, 0, 1), pathspace, range_remapping)


def resample_to_reso(v, tgt_shape):
    if len(v.shape) < len(tgt_shape)+2:
        return resample_to_reso(v.unsqueeze(0), tgt_shape).squeeze(0)
    elif v.shape[2:] == tgt_shape:
        return v
    else:
        assert(len(tgt_shape)==2), f"{v.shape=}, {tgt_shape=}"
        return torch.nn.functional.interpolate( v, size=tgt_shape
                                              , mode='bilinear', align_corners=False )
        

class IdentityRemapping(RangeRemapping):
    def __init__(self):
        pass
    def to_unitinterval(self, x):
        return x
    def from_unitinterval(self, y):
        return y

class SigmoidRemapping(RangeRemapping):
    def __init__(self):
        pass
    def to_unitinterval(self, x):
        return torch.sigmoid(x)
    def from_unitinterval(self, y):
        # https://stackoverflow.com/a/66116934/745903
        return -torch.log(torch.reciprocal(y) - 1)

class SelectableHardnessClipping(RangeRemapping):
    def __init__(self, hardness=2):
        self.hardness=hardness
    def from_unitinterval(self, y):
        return y - (1/y + 1/(y-1))/self.hardness
    def to_unitinterval(self, x):
        # x = y − (1/y + 1/(y−1))/h
        # x·h·y·(y-1) = h·y²·(y−1) − (y−1) − y
        # x·h·y² − x·h·y − h·y³ + h·y² + 2·y − 1 = 0
        # y³ − (x+1)·y² + (x−2/h)·y + 1/h = 0
        # y =: t + (x+1)/3
        # y² = t² + ⅔·t·(x+1) + (x+1)²/9
        # y³ = t³ + t²·(x+1) + t·(x+1)²/3 + (x+1)³/27
        # t³ + t²·(x+1) + t·(x+1)²/3 + (x+1)³/27
        #  − (x+1)·(t² + ⅔·t·(x+1) + (x+1)²/9)
        #  + (x−2/h)·(t + (x+1)/3) + 1/h = 0
        # t³ − (x+1)²·t/3 − 2/27·(x+1)³
        #  + (x−2/h)·t + (x−2/h)·(x+1)/3 + 1/h = 0
        # 0 = t³
        #      + (x − (x+1)²/3 − 2/h)·t              } p
        #      + (x−2/h)·(x+1)/3 − 2/27·(x+1)³ + 1/h } q
        h = self.hardness
        p = x - (x+1)**2/3 - 2/h
        q = (x-2/h)*(x+1)/3 - 2/27*(x+1)**3 + 1/h
        # w := √(-p/3)
        w = torch.sqrt(-p/3)
        t = 2*w*torch.cos(torch.acos(3*q/(2*p*w))/3 - 2*np.pi/3)
        return t + (x+1)/3

class SelectableHardnessSigmoid(RangeRemapping):
    def __init__(self, hardness=2):
        self.hardness=hardness
    def from_unitinterval(self, y):
        z = 2*y - 1   # Change range to [-1,1]
        return z + z / (self.hardness * (1 - z**2))
    def to_unitinterval(self, x):
        # x = z + z/(h·(1−z²))
        # h·x·(1-z²) = z·h·(1−z²) + z
        # h·x − h·x·z² = z·h − h·z³ + z
        # h·z³ − h·x·z² − z·(1 + h) + h·x = 0
        # z =: t + x/3
        # h·(t + x/3)³ − h·x·(t + x/3)² − (t + x/3)·(1+h) + h·x = 0
        # h·t³ + h·x·t² + h·x²·t/3 + h·x³/27
        #              − h·x·t² − ⅔·h·x²·t − h·x³/9
        #                               − t − h·t − x/3 − h·x/3
        #                                                 + h·x = 0
        # h·t³ − (h·x²/3 + 1 + h)·t − h·x³·2/27 − x/3 + ⅔·h·x = 0
        # t³ − (x²/3 + 1/h + 1)·t − x³·2/27 − x/3h + ⅔·x = 0
        # p := -(x²/3 + 1/h + 1)
        p = -(x**2/3 + 1/self.hardness + 1)
        # q := -x³·2/27 − x/3h + ⅔·x
        q = -x**3*2/27 - x/(3*self.hardness) + 2/3*x
        # w := √(-p/3)
        w = torch.sqrt(-p/3)
        # t = 2·w·cos(⅓·acos(3·q/(2·p·w)) − ⅔·π)
        # (Viète 2006, doi:10.1017/S0025557200179598)
        t = 2*w*torch.cos(torch.acos(3*q/(2*p*w))/3 - 2*np.pi/3)
        z = t + x/3
        return (z+1) / 2

class OptstepStrategy(ABC):
    @abstractmethod
    def factor_for_update(self, update):
        return NotImplemented

class ConstFactorOptStep(OptstepStrategy):
    def __init__(self, const_update_factor):
        self.const_update_factor = const_update_factor
    def factor_for_update(self, update):
        return self.const_update_factor

class LInftyNormalizingOptStep(OptstepStrategy):
    def __init__(self, update_supremum):
        self.update_supremum = update_supremum
    def factor_for_update(self, update):
        norm = float(torch.max(torch.abs(update)))
        return self.update_supremum/norm

class AblPathObjective(Enum):
    FwdLongestRetaining=0
    BwdQuickestDissipating=1
    FwdRetaining_BwdDissipating=2
    BoundaryStraddle_Contrast=3
    FwdQuickestDissipating=4

class LinInterpPathsSpace(PathsSpace):
    def __init__(self, x, baseline):
        assert(x.shape == baseline.shape)
        self.x = x
        self.baseline = baseline
        self.difference = baseline - x
    @property
    def ablmask_shape(self):
        return self.x.shape[1:]
    def apply_mask_seq(self, abl_seq: torch.Tensor):
        if abl_seq.shape<=self.x.shape:
            abl_seq = abl_seq.unsqueeze(1)
        assert(abl_seq.shape[1]==1 and self.x.shape[1:] == abl_seq.shape[2:]
              ), f"{abl_seq.shape=}, {self.x.shape=}"
        return self.x + self.difference * abl_seq
    @property
    def target_image(self):
        return self.x
    @property
    def baseline_image(self):
        return self.baseline

class BlurPyramidSigmaInterp(Enum):
    Linear=0
    Logarithmic=1

class BlurPyramidPathsSpace(PathsSpace):
    """A space/pertubation in which mask-values correspond to how strongly
    each region of the image is blurred out. In other words, this is a lowpass
    filter with position-dependent cutoff. It corresponds to the
    `BLUR_PERTURBATION` option of the TorchRay `Pertubation` class."""
    def __init__( self, x, num_levels=8, max_blur=20, min_blur=0
                , sigma_interp=BlurPyramidSigmaInterp.Linear ):
        nCh,h,w = x.shape[-3:]
        self.x = x.reshape(nCh,h,w)

        if sigma_interp is BlurPyramidSigmaInterp.Logarithmic and min_blur==0:
            min_blur = max_blur / 2**(num_levels/2)

        σ_interp = {
            BlurPyramidSigmaInterp.Linear:
                lambda η: min_blur + η*(max_blur-min_blur)
          , BlurPyramidSigmaInterp.Logarithmic:
                lambda η: min_blur * np.exp(η*np.log(max_blur/min_blur))
          }[sigma_interp]
            
        self.pyramid = torch.cat(
              [ apply_filter( self.x.unsqueeze(0), σ_interp(float(1-s)) )
               for s in torch.linspace(0, 1, num_levels) ]
            , dim=0 ).flip(0)
        self.pyramid.requires_grad = False

    @property
    def ablmask_shape(self):
        return self.x.shape[1:]

    def apply_mask_seq(self, abl_seq: torch.Tensor):
        # Adapted from TorchRay,
        # https://github.com/facebookresearch/TorchRay/blob/6a198ee/torchray/attribution/extremal_perturbation.py#L156
        n = abl_seq.shape[0]
        w = abl_seq.reshape(n, 1, *abl_seq.shape[1:])
        num_levels = self.pyramid.shape[0]
        w = w * (num_levels - 1)
        k = w.floor()
        w = w - k
        k = k.long()

        y = self.pyramid[None, :]
        y = y.expand(n, *y.shape[1:])
        k = k.expand(n, 1, *y.shape[2:])
        y0 = torch.gather(y, 1, k)
        y1 = torch.gather(y, 1, torch.clamp(k + 1, max=num_levels - 1))

        return ((1 - w) * y0 + w * y1).squeeze(dim=1)

    @property
    def target_image(self):
        return self.x
    @property
    def baseline_image(self):
        return self.pyramid[0]

class MaskJitter(ABC):
    """This class applies a modification to a mask-path, possibly
    involving a random-generator state change. Unlike with `RangeRemapping`,
    the modification is not undone after model evaluation in order to
    perform a gradient-descent step; instead the gradients on the modified form
    are simply applied to the state in its original form.
    This is reasonable if the modification only applied a constant or
    independently-random offset to the signal. The intended purpose is
    to inject stochasticity and/or quantize the mask values that actually
    appear for the interpolation routine."""
    @abstractmethod
    def jitter_mask(self, abl_seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class NOPMaskJitter(MaskJitter):
    def __init__(self):
        pass
    def jitter_mask(self, abl_seq: torch.Tensor) -> torch.Tensor:
        return abl_seq

class GaussianJitter(MaskJitter):
    def __init__( self, jitter_stddev: float = 0.5
                      , jitter_filtering: FilteringConfig = NOPFilteringConfig()
                      , rng: Optional[torch.Generator] = None ):
        self.jitter_stddev = jitter_stddev
        self.jitter_filtering = jitter_filtering
        self.rng = rng
        if rng is None:
            self.rng = torch.Generator()
            self.rng.manual_seed(17584640331630194775)
        else:
            self.rng = rng
    def jitter_mask(self, abl_seq: torch.Tensor) -> torch.Tensor:
        disturbance = apply_filter( torch.normal( 0.0, 1.0, size=abl_seq.shape
                                                , generator=self.rng, requires_grad=False
                                                ).to(abl_seq.device)
                                  , self.jitter_filtering )
        disturbance_norm = np.sqrt(
           float(torch.sum(disturbance**2)/len(torch.flatten(abl_seq))) )
        return abl_seq + disturbance * (self.jitter_stddev / disturbance_norm)

class HardQuantizedMasks(MaskJitter):
    """Take masks as _probabiliies_ (i.e. floats in range 0 to 1)
    and yield boolean masks. If `rng` is provided, the boolean values
    will be random, namely 1 with probability corresponding to the float
    value contained in the original mask. If no generator is provided,
    simply all values below 0.5 are pulled to 0, all above 0.5 to 1."""
    def __init__( self, prejitter: MaskJitter = NOPMaskJitter()
                      , quantize_threshold: Optional[Number] = 0.5
                      , rng: Optional[torch.Generator] = None ):
        self.prejitter = prejitter
        self.quantize_threshold = quantize_threshold
        self.rng = rng
    def jitter_mask(self, abl_seq: torch.Tensor) -> torch.Tensor:
        jittered = self.prejitter.jitter_mask(abl_seq).clone().detach()
        if self.rng is None:
            threshold = self.quantize_threshold
        else:
            threshold = ( self.quantize_threshold
                         + torch.rand(abl_seq.shape, generator=self.rng)
                         - 0.5 ).to(abl_seq.device)
        jittered[jittered<threshold] = 0
        jittered[jittered>0] = 1
        return jittered

def mk_suitable_label(label_nr, model, x, baseline=None):
    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))
    elif label_nr=='baseline_label':
        assert(baseline is not None)
        label_nr = torch.argmax(model(baseline.unsqueeze(0)))
    return label_nr

class GradEstimation_Strategy(Enum):
    autodiff_grad = 0
    jitterstochastic_finite_diff = 1

def gradientMove_ablation_path( model, pathspace, abl_seq
                              , optstep
                              , objective=AblPathObjective.FwdLongestRetaining
                              , label_nr=None
                              , pointwise_scalar_product=True
                              , gradients_postproc=lambda gs: gs
                              , range_remapping = IdentityRemapping()
                              , mask_jitter = NOPMaskJitter()
                              , grad_estim_strategy = GradEstimation_Strategy.autodiff_grad
                              ):
    nSq = abl_seq.shape[0]
    do_boundarystraddle = objective is AblPathObjective.BoundaryStraddle_Contrast
    mask_shape = abl_seq.shape[2:] if do_boundarystraddle else abl_seq.shape[1:]
    x = pathspace.target_image
    baseline = pathspace.baseline_image
    nCh = x.shape[0]

    needs_resampling = mask_shape != pathspace.ablmask_shape

    label_nr = mk_suitable_label(label_nr, model, x, baseline)

    # Suitably reshaped and resampled version of mask, for applying (with
    # auto-broadcast channel dimension) to target- and baseline images.
    resampled_abl_seq = resample_to_reso(
             abl_seq.reshape(nSq, 2 if do_boundarystraddle else 1, *mask_shape)
           , pathspace.ablmask_shape
           )

    # If given a suitable remapping function, use it for a representation
    # of the ablation path that is not limited to the range [0,1].
    delimited_abl_seq = range_remapping.from_unitinterval(resampled_abl_seq)

    assert(pointwise_scalar_product
      ), "Optimising without pointwise scalar product currently not supported."

    def evalmodel_quantizedmasked():
        quantized_abl_seq = mask_jitter.jitter_mask(delimited_abl_seq)

        if grad_estim_strategy is GradEstimation_Strategy.autodiff_grad:
            quantized_abl_seq.requires_grad = True

        quantized_relimited = range_remapping.to_unitinterval(quantized_abl_seq)

        argument = pathspace.apply_mask_seq(
                               { AblPathObjective.FwdLongestRetaining:
                                    lambda rabs: rabs
                               , AblPathObjective.FwdQuickestDissipating:
                                    lambda rabs: rabs
                               , AblPathObjective.BwdQuickestDissipating:
                                    lambda rabs: 1-rabs
                               , AblPathObjective.FwdRetaining_BwdDissipating:
                                    lambda rabs: torch.cat([rabs, 1-rabs], dim=0)
                               , AblPathObjective.BoundaryStraddle_Contrast:
                                    lambda rabs: torch.cat([rabs[:,0], rabs[:,1]], dim=0).unsqueeze(1)
                               }[objective]
                                (quantized_relimited)
                       )
 
        n_positive_weighted = ( 0 if objective in [ AblPathObjective.BwdQuickestDissipating
                                                  , AblPathObjective.FwdQuickestDissipating ]
                               else nSq )
        n_negative_weighted = ( 0 if objective is AblPathObjective.FwdLongestRetaining
                               else nSq )
 
        model_result = model(argument)
        n_evals, n_classes = model_result.shape[:2]
        assert(n_evals == n_positive_weighted+n_negative_weighted
              ), f"{n_positive_weighted=}, {n_negative_weighted=}, {n_evals=}"
 
        eval_signs = torch.tensor( [ 1 for _ in range(n_positive_weighted)]
                                 + [-1 for _ in range(n_negative_weighted)]
                                 ).to(abl_seq.device)

        # Ablation path score, computed as the "integral": average of the
        # target-class probability over the path. Backward-dissipating contributions
        # are weighed negatively.
        intg_score = torch.mean(
                        torch.softmax(model_result.reshape(n_evals,n_classes), -1)
                                           [:, label_nr]
                            * eval_signs
                      ) * (2 if objective is AblPathObjective.FwdRetaining_BwdDissipating
                                 or do_boundarystraddle
                            else 1
                      ) + (1 if objective is AblPathObjective.FwdQuickestDissipating
                            else 0)
 
        if grad_estim_strategy is GradEstimation_Strategy.autodiff_grad:
            # Gradient of the integral-score, as a function of the entire mask-path;
            # in shape of its oversampled form.
            grad = pathspace.differential_to_gradient(
                                torch.autograd.grad( intg_score
                              , quantized_abl_seq )[0] )
           
            grad = gradients_postproc(grad)
            
            return intg_score, grad
        
        else:
            return quantized_relimited, intg_score

    if grad_estim_strategy is GradEstimation_Strategy.autodiff_grad:
        intg_score, grad = evalmodel_quantizedmasked()
    elif grad_estim_strategy is GradEstimation_Strategy.jitterstochastic_finite_diff:
        with torch.no_grad():
            q0, intg_score0 = evalmodel_quantizedmasked()
            q1, intg_score1 = evalmodel_quantizedmasked()
            distance = torch.linalg.vector_norm(q1 - q0)**2
            if not (distance > 0):
                print(q0)
                print(q1)
                raise ZeroDivisionError(f"{distance=}")
            grad = (q1 - q0) * (intg_score1 - intg_score0) / distance
        intg_score = (intg_score0 + intg_score1) / 2
    else:
        raise ValueError(f"Unknown gradient-estimation strategy {grad_estim_strategy}")

    if optstep is None:
        raise NotImplementedError("Automatic opt-step selection")
    elif isinstance(optstep, float):
        update = grad * optstep
    elif isinstance(optstep, OptstepStrategy):
        update = grad * optstep.factor_for_update(grad)
    assert(update.shape==(nSq, 2 if do_boundarystraddle else 1, *pathspace.ablmask_shape)
          ), f"{update.shape=}, {nSq=}, {pathspace.ablmask_shape=}"
    
    resampled_abl_seq = range_remapping.to_unitinterval(
                delimited_abl_seq + update
                           ).detach()

    if do_boundarystraddle:
        abl_seq[:] = resample_to_reso(resampled_abl_seq, mask_shape)
    else:
        abl_seq[:] = resample_to_reso(resampled_abl_seq, mask_shape)[:,0]

    return float(intg_score)

def saturated_masks(φ, saturation):
    saturation = saturation + 1e-6 # Avoid NaN from zero saturation
                                   # causing 0/0 division.
    return (torch.tanh( (φ*2 - torch.ones_like(φ))*saturation )
                          / (torch.tanh(torch.tensor(saturation)))
                       + torch.ones_like(φ))/2

class SaturationAdjustment(ABC):
    @abstractmethod
    def adjust_mask_saturation(self, masks: torch.Tensor, i_optstep: int):
        raise NotImplementedError

class SaturationBoost(SaturationAdjustment):
    def __init__(self, saturation: float):
        self.saturation = saturation
    def adjust_mask_saturation(self, masks, i_optstep):
        return saturated_masks(masks, self.saturation)

class StepcountDepSaturationAdjustment(SaturationAdjustment):
    def __init__(self, sdep_adj: Callable[[int], SaturationAdjustment]):
        self.sdep_adj = sdep_adj
    def adjust_mask_saturation(self, masks, i_optstep):
        return self.sdep_adj(i_optstep
                  ).adjust_mask_saturation(masks, i_optstep)

class BorderVanish_SaturationBoost(SaturationAdjustment):
    def __init__(self, saturation: float):
        self.saturation = saturation
    def adjust_mask_saturation(self, masks, i_optstep):
        saturation_window = bordervanish_window(masks)
        return saturated_masks(masks, self.saturation*saturation_window)


def boundarystraddle_shrinkwrap(path, wrap_strength=0.1):
    """A “filter” to be used on the pairs of paths involved in
    `AblPathObjective.BoundaryStraddle_Contrast`. It does not affect
    the masks directly but instead pulls both of the paths closer
    together, and specifically in a way that causes the highlighted
    path to contain masks that are preferrably only locally more active
    than the contrasting path."""
    path_steps = path.shape[0]
    assert(path.shape[1]==2
          ), f"{path.shape=}"
    def ctr_attract(x):
        return x*(1-wrap_strength) + x**2*wrap_strength
    return torch.stack( [ path[:,0]
                        , path[:,0] + ctr_attract(path[:,1] - path[:,0]) ]
                      , dim=1 )

def path_optimisation_sequence (
          model, pathspaces, path_steps, optstep
        , saturation=0, filter_cfg=None, filter_mix_ratio=1
        , initpth=None, ablmask_resolution=None
        , pathrepairer=repair_ablation_path
        , objective=AblPathObjective.FwdLongestRetaining
        , boundaryshrink_wrap_strength=None
        , momentum_inertia=0
        , range_remapping: RangeRemapping =IdentityRemapping()
        , **kwargs ):
    example_pathspace = pathspaces()
    x_example = example_pathspace.target_image
    img_shape = x_example.shape[1:]
    if ablmask_resolution is None:
        ablmask_resolution = example_pathspace.ablmask_shape

    if objective is AblPathObjective.BoundaryStraddle_Contrast:
        if boundaryshrink_wrap_strength is None:
            boundaryshrink_wrap_strength = 0.1
    else:
        assert(boundaryshrink_wrap_strength is None)

    ablmask_slice_shape = (
            (2,)+ablmask_resolution
                if objective is AblPathObjective.BoundaryStraddle_Contrast
                else ablmask_resolution )

    pth = ( torch.stack([p*torch.ones(ablmask_slice_shape)
                         for p in np.linspace(0,1,path_steps)[1:-1]])
                   .to(x_example.device)
             if initpth is None else initpth )
    if momentum_inertia>0:
        momentum = torch.zeros_like(pth)

    for i in count():
        old_pth = pth.clone().detach()

        pathspace = pathspaces()

        current_score = gradientMove_ablation_path(
              model, pathspace, abl_seq=pth, optstep=optstep
            , objective=objective
            , range_remapping=range_remapping, **kwargs )
        yield old_pth, current_score

        if momentum_inertia>0:
            momentum = ( momentum * momentum_inertia
                        + (pth - old_pth)*(1-momentum_inertia) )
            pth = old_pth + momentum

        if isinstance(saturation, SaturationAdjustment):
            pth = saturation.adjust_mask_saturation(pth, i)
        elif isinstance(saturation, Number) and saturation>0:
            pth = saturated_masks(pth,saturation)

        def filterWith(σ):
            if ablmask_resolution is not None:
                scale_factor = np.sqrt( np.product(ablmask_resolution)
                                       / np.product(img_shape) )
            nonlocal pth
            pth = ( pth*(1-filter_mix_ratio)
                   + apply_filter(pth, σ.rescaled(scale_factor))*filter_mix_ratio )
        if isinstance(filter_cfg, NOPFilteringConfig):
            pass
        elif callable(filter_cfg):
            filterWith(filter_cfg(i))
        elif filter_cfg is not None:
            filterWith(filter_cfg)

        if objective is AblPathObjective.BoundaryStraddle_Contrast:
            pth = boundarystraddle_shrinkwrap(pth, wrap_strength=boundaryshrink_wrap_strength)

        pth = pathrepairer(pth, pathspace=pathspace, range_remapping=range_remapping)
        
        if momentum_inertia>0:
            momentum = pth - old_pth

        assert(not(torch.any(torch.isnan(pth))))

class PathOptFinishCriterion(ABC):
    def __init__(self, subcriteria_dnf):
        self._subcriteria_dnf = subcriteria_dnf

    def __call__(self, iterations_done, abl_path, score):
        return any([all([c(iterations_done, abl_path, score)
                          for c in cs])
                     for cs in self._subcriteria_dnf])

    def _as_criteria_dnf(self):
        return ( self._subcriteria_dnf
                if (type(self) is PathOptFinishCriterion)
                else [[self]] )

    def _as_criteria_conjunction(self):
        return ( self._subcriteria_dnf[0]
                if (type(self) is PathOptFinishCriterion
                     and len(self._subcriteria_dnf)==1)
                else [self] )

    def __or__(self, other):
        return PathOptFinishCriterion(
           self._as_criteria_dnf() + other._as_criteria_dnf() )

    def __and__(self, other):
        return PathOptFinishCriterion(
           [self._as_criteria_conjunction() + other._as_criteria_conjunction()] )

class FixedStepCount(PathOptFinishCriterion):
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations

    def __call__(self, iterations_done, abl_path, score):
        return iterations_done >= self.n_iterations

def path_saturation(abl_path):
    return torch.mean(2 * torch.abs(abl_path - 0.5));

class SaturationTarget(PathOptFinishCriterion):
    def __init__(self, saturation_target):
        self.saturation_target = saturation_target

    def __call__(self, iterations_done, abl_path, score):
        return float(path_saturation(abl_path)
                    ) >= self.saturation_target

@dataclass
class AblPathStateSummary:
    path: torch.Tensor
    score: float
    saturation: float

@dataclass
class AblPathOptimProgressShower:
    summary_show: Callable[[AblPathStateSummary], str]
    overwrite_same_line: bool = True
    def __call__(self, summary: AblPathStateSummary):
        return self.summary_show(summary)
    def __bool__(self):
        return True

def optimised_path( model, x=None, baselines=None
                  , path_steps=16, optstep=LInftyNormalizingOptStep(0.5)
                  , finish_criterion= FixedStepCount(6) &
                                       (SaturationTarget(0.8) | FixedStepCount(100))
                  , pathspaces=None
                  , logging_destination=None
                  , progress_on_stdout=False
                  , **kwargs):
    if pathspaces is None:
        assert(x is not None)
        if callable(baselines):
             pathspaces = lambda: LinInterpPathsSpace(x, baselines())
        elif isinstance(baselines, torch.Tensor):
             pathspace = LinInterpPathsSpace(x, baselines())
             pathspaces = lambda: pathspace
        else:
             raise TypeError(f"Unsupported {type(baselines)=}")
    else:
        assert(x is None and baselines is None)
        if isinstance(pathspaces, PathsSpace):
            const_pathspace = pathspaces
            pathspaces = lambda: const_pathspace

    i = 0

    for pth, current_score in path_optimisation_sequence (
          model, pathspaces, path_steps, optstep, **kwargs ):
        sat = float(path_saturation(pth))
        state_summary = AblPathStateSummary(
                         path=pth, score=current_score, saturation=sat)
        if finish_criterion(i, pth, current_score):
            return state_summary
        if logging_destination is not None:
            print(current_score, file=logging_destination, flush=True)
        if progress_on_stdout:
            if isinstance(progress_on_stdout, AblPathOptimProgressShower):
                print( progress_on_stdout(state_summary)
                     , end=("\r" if progress_on_stdout.overwrite_same_line else "\n") )
            else:
                print(f"saturation: {sat:.2f}, score: {current_score:.3g}        ", end="\r")
        i+=1

def masked_interpolation(x=None, baseline=None, abl_seq=None, pathspace=None, include_endpoints=False):

    if pathspace is None:
        assert(x is not None and baseline is not None)
        pathspace = LinInterpPathsSpace(x, baseline)
    else:
        assert(x is None and baseline is None)
        x = pathspace.target_image
        baseline = pathspace.baseline_image

    needs_resampling = pathspace.ablmask_shape != abl_seq.shape[1:]
    if type(abl_seq) != torch.Tensor:
        abl_seq = torch.stack(list(abl_seq))
    xOpt = x.to(abl_seq.device)
    nSq = abl_seq.shape[0]
    mask_shape = abl_seq.shape[1:]
    nCh = x.shape[0]

    rspl_seq = resample_to_reso( abl_seq.reshape(nSq, 1, *mask_shape)
                               , pathspace.ablmask_shape
                ) if needs_resampling else abl_seq

    if include_endpoints:
        rspl_seq = torch.concat( [ torch.zeros_like(rspl_seq[0]).unsqueeze(0)
                                 , rspl_seq
                                 , torch.ones_like(rspl_seq[-1]).unsqueeze(0) ]
                               , dim=0 )

    return pathspace.apply_mask_seq(rspl_seq)

def most_salient_mask_in_path( abl_seq, model, x=None, baseline=None, pathspace=None
                             , minimum_ablation_pos=0.25, label_nr=None
                             , fallback_to_best_scoring=True ):
    if len(abl_seq) <= 1:
        return 0

    if pathspace is None:
        pathspace = LinInterpPathsSpace(x, baseline)
    else:
        x = pathspace.target_image

    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))

    predictions = model(masked_interpolation(abl_seq=abl_seq, pathspace=pathspace))
    imax = len(abl_seq) - 1
    min_allowed_i = int(len(abl_seq) * minimum_ablation_pos)
    while imax>=min_allowed_i and torch.argmax(predictions[imax])!=label_nr:
        imax -= 1
    if imax < min_allowed_i:
        if not fallback_to_best_scoring:
            return None
        best_score = -np.inf
        imax = min_allowed_i
        for i in range(min_allowed_i, len(abl_seq)):
            score = torch.softmax(predictions[i], 0)[label_nr]
            if score > best_score:
                best_score = score
                imax = i
    return imax

# This function only considers transitions from the requested class to another one.
def find_class_transition( model, x, baseline, abl_seq, **kwargs ):
    return most_salient_mask_in_path( abl_seq, model, x, baseline
                                    , fallback_to_best_scoring=False
                                    , **kwargs )


def bordervanish_window(m):
    if isinstance(m, torch.Tensor):
        ys, xs = torch.meshgrid( torch.linspace(0, np.pi, m.shape[-2])
                                                , torch.linspace(0, np.pi, m.shape[-1]) )
        window = torch.sqrt(torch.clamp(torch.sin(xs) * torch.sin(ys), min=0, max=1)
                                                  ).to(m.device)
        return m * window
    elif isinstance(m, np.ndarray):
        return bordervanish_window(torch.tensor(m)).numpy()
    else:
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(m)}")

def standard_grad_postproc( filter_conf = FilteringConfig(None)
                          , suppress_borders = False
                          , remove_const_bias = True ):
    def gpp(g):
        if suppress_borders:
            g = bordervanish_window(g)
        if remove_const_bias:
            g -= torch.mean(g, dim=(-2,-1), keepdim=True)
        return apply_filter(g, filter_conf)
    return gpp

def quick_tensor_info(t):
    return f"shape: {t.shape}, min: {torch.min(t)}, max: {torch.max(t)}"

def influence_weighted_increment_saliency(
                 abl_seq, model, x, baseline
               , label_nr=None ):
    assert(len(abl_seq) >= 1)
    
    abl_seq = torch.concat([ torch.zeros_like(abl_seq[0]).unsqueeze(0)
                           , abl_seq
                           , torch.ones_like(abl_seq[0]).unsqueeze(0) ])

    predictions = torch.softmax(model(
                     masked_interpolation( x,baseline,abl_seq )), 1).detach()
    
    if label_nr is None:
        label_nr = torch.argmax(predictions[0])

    influences = predictions[:-1, label_nr] - predictions[1:, label_nr]
    increments = abl_seq[1:] - abl_seq[:-1]
    
    # We have two differentials and one integration, so need to divide once
    # by the size of the steps (or equivalenty, multiply once by their number)
    return increments.shape[0] * torch.sum(influences * increments, axis=0)
