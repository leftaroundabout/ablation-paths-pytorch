##############################################################################
#
# Copyright 2020 Olivier Verdier, Justus Sagemüller
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

from __future__ import annotations

import numpy as np
import numbers
import torch

import scipy as sp
from scipy.interpolate import interp1d

import scipy.ndimage

import torch.fft

import odl

from abc import ABC, abstractmethod
from enum import Enum

from typing import Optional


def img_fft(image):
    return torch.fft.fft(torch.fft.fft(image, dim=1), dim=2)
def img_ifft(spectrum):
    return torch.fft.ifft(torch.fft.ifft(spectrum, dim=2), dim=1)

def freq_range(dimlen):
    halvlen = dimlen//2
    return list(range(0,dimlen - halvlen)) + list(range(-halvlen,0))

def apply_brickwall_filter(image, sigma):
    nCh, w, h = image.shape
    spectrum = img_fft(image)
    spectr_mask = torch.tensor(
                      [ [ [1 if (ν/w)**2 + (μ/h)**2 < 1/(np.pi*sigma)**2 else 0
                           for μ in freq_range(h)]
                         for ν in freq_range(w)]
                       for chn in range(nCh)])
    return torch.real(img_ifft(spectrum * spectr_mask.to(image.device)))

def apply_gaussian_filter(image, sigma):
    assert(isinstance(sigma, numbers.Number))
    filtered = [sp.ndimage.filters.gaussian_filter(monochromatic, sigma=sigma)
                        for monochromatic in image.cpu()]
    return torch.Tensor(filtered).to(image.device)

def get_sobolev_metric(space, scale=1.):
    lap = odl.discr.diff_ops.Laplacian(space, pad_mode='order0')
    op = odl.operator.IdentityOperator(space) - scale**2 * lap
    return op

def border_stretch_trafo(image, axis=None, interp_kind='cubic', undo=False):
    if isinstance(image, torch.Tensor):
        return torch.tensor( border_stretch_trafo (image.cpu().numpy()
                                                 , axis=axis, interp_kind=interp_kind, undo=undo )
                           , dtype=image.dtype
                           ).to(image.device)
    if axis is None:
        for i in range(len(image.shape)):
            image = border_stretch_trafo(image, axis=i, interp_kind=interp_kind, undo=undo)
        return image
    else:
        xs = np.linspace(-np.pi/2, np.pi/2, image.shape[axis], dtype=image.dtype)
        xs_remapped = np.sin(xs)*np.pi/2
        if undo:
            xs, xs_remapped = xs_remapped, xs
        return interp1d(xs, image, kind=interp_kind, axis=axis)(xs_remapped)

def apply_borderstretched_gaussian_filter(image, sigma):
    return border_stretch_trafo(apply_gaussian_filter( border_stretch_trafo(image)
                                                     , sigma=sigma
                                                     ), undo=True )

def apply_sobolevdualproj_filter(image, scale):
    if type(image) is odl.DiscretizedSpaceElement:
        space = image.space
        φreg = space.one()
        odl.solvers.iterative.conjugate_gradient(
              get_sobolev_metric(space, scale), φreg, image, niter=40)
        return φreg.copy()
    elif type(image) is np.ndarray:
        d = len(image.shape)
        space = odl.discr.uniform_discr(
            min_pt=list(0 for _ in range(d))
          , max_pt=list(1 for _ in range(d))
          , shape=image.shape
          , dtype='float32' )
        image_odl = space.element(image)
        return apply_sobolevdualproj_filter(image_odl, scale).asarray()
    elif type(image) is torch.Tensor:
        return torch.Tensor(
                  apply_sobolevdualproj_filter(image.cpu().numpy(), scale)
                ).to(image.device)
    else:
        raise TypeError("Supports only odl.DiscretisedSpaceElement, numpy.ndarray and torch.Tensor")

class AbstractFilteringConfig(ABC):
    @abstractmethod
    def rescaled(scl_factor) -> AbstractFilteringConfig:
        raise NotImplementedError
    @property
    def filter_dimensionality(self) -> Optional[int]:
        """Filters act on tensors, specifically on the last n dimensions.
        This property describes how many those are. If `None`, this means
        it depends on the input (in a way that is not generally specified)."""
        return None

class NOPFilteringConfig(AbstractFilteringConfig):
    def __init__(self):
        pass
    def rescaled(self, scl_factor):
        return self

class CustomFilteringConfig(AbstractFilteringConfig):
    def __init__(self, custom_filter_fn, scl_factor=1.0, dimensionality=None):
        self.custom_filter_fn = custom_filter_fn
        self.scl_factor = scl_factor
        self.dimensionality = dimensionality
    def rescaled(self, scl_factor):
        return CustomFilteringConfig( self.custom_filter_fn
                                    , scl_factor=self.scl_factor*scl_factor )
    @property
    def filter_dimensionality(self) -> Optional[int]:
        return self.dimensionality

class LowpassFilterType(Enum):
    Gaussian=1
    BorderStretched_Gaussian=2
    Brickwall=3

class LowpassFilteringConfig(AbstractFilteringConfig):
    def __init__(self, filter_type, sigma, dimensionality=2):
        self.filter_type = filter_type
        self.sigma = ( sigma.sigma if isinstance(sigma, FilteringConfig)
                         else sigma )
        self.dimensionality = dimensionality
    def rescaled(self, scl_factor):
        assert(isinstance(scl_factor, numbers.Number))
        return FilteringConfig(self.filter_type, self.sigma * scl_factor)
    @property
    def filter_dimensionality(self) -> Optional[int]:
        return self.dimensionality

class SymmetrizingFilterType(Enum):
    TimeRev_Is_OppositeMask=1
    TimeRev_Is_Negative=2

class SymmetrizeFilteringConfig(AbstractFilteringConfig):
    def __init__(self, filter_type):
        self.filter_type = filter_type
    def rescaled(self, _):
        return self

class FilteringConfig(AbstractFilteringConfig):
    def __init__(self, filter_type=None, sigma=None, filters_pipeline=None):
        if isinstance(filter_type, LowpassFilterType):
            assert(isinstance(sigma, numbers.Number))
            assert(filters_pipeline is None)
            self.filters_pipeline = [LowpassFilteringConfig(filter_type, sigma=sigma)]
        elif isinstance(filter_type, SymmetrizingFilterType):
            assert(sigma is None)
            self.filters_pipeline = [SymmetrizeFilteringConfig(filter_type)]
        else:
            assert(sigma is None)
            assert(filter_type is None)
            if filters_pipeline is not None:
                for ftr in filters_pipeline:
                    assert(isinstance(ftr, AbstractFilteringConfig))
                self.filters_pipeline = filters_pipeline
            else:
                self.filters_pipeline = []
    def rescaled(self, scl_factor):
        return FilteringConfig(filters_pipeline
                  = [ftr.rescaled(scl_factor) for ftr in self.filters_pipeline])

class ComplementaryFilteringConfig(AbstractFilteringConfig):
    def __init__(self, opposite_filter: AbstractFilteringConfig):
        self.opposite_filter = opposite_filter
    def rescaled(self, scl_factor):
        return ComplementaryFilteringConfig(
                 opposite_filter = self.opposite_filter.rescaled(scl_factor))

def apply_filter(image, ftr_conf=6):
    def on_smashed_bash_dims(f):
        d = ftr_conf.filter_dimensionality
        def smashed_f(x):
            if d is not None:
                batch_dims = x.shape[ : -d]
                filter_dims = x.shape[-d : ]
                return f(x.reshape(np.product(batch_dims), *filter_dims)
                        ).reshape(*batch_dims, *filter_dims)
            else:
                return f(x)
        return smashed_f

    if isinstance(ftr_conf, NOPFilteringConfig):
        return image
    elif isinstance(ftr_conf, CustomFilteringConfig):
        return ftr_conf.custom_filter_fn(image, scl_factor=ftr_conf.scl_factor)
    elif isinstance(ftr_conf, LowpassFilteringConfig):
        return on_smashed_bash_dims({
            LowpassFilterType.Gaussian: lambda img:
               apply_gaussian_filter( img, ftr_conf.sigma )
          , LowpassFilterType.Brickwall: lambda img:
               apply_brickwall_filter(img, ftr_conf.sigma)
          , LowpassFilterType.BorderStretched_Gaussian: lambda img:
               apply_borderstretched_gaussian_filter( img, ftr_conf.sigma
                               , dimensionality=ftr_conf.filter_dimensionality )
          }[ftr_conf.filter_type])(image)
    elif isinstance(ftr_conf, SymmetrizeFilteringConfig):
        return {
       SymmetrizingFilterType.TimeRev_Is_OppositeMask: lambda imgs:
          (imgs + 1 - torch.flip(imgs, dims=((-3,)))) / 2
     , SymmetrizingFilterType.TimeRev_Is_Negative: lambda imgs:
          (imgs - torch.flip(imgs, dims=((-3,)))) / 2
     }[ftr_conf.filter_type](image)
    elif isinstance(ftr_conf, FilteringConfig):
        for iftr in ftr_conf.filters_pipeline:
            image = apply_filter(image, iftr)
        return image
    elif isinstance(ftr_conf, ComplementaryFilteringConfig):
        ofiltrd = apply_filter(image, ftr_conf.opposite_filter)
        return image - ofiltrd
    else:
        assert(isinstance(ftr_conf, numbers.Number)), f"{type(ftr_conf)}"
        return apply_filter(image
                , ftr_conf = FilteringConfig(LowpassFilterType.Gaussian, ftr_conf) )
