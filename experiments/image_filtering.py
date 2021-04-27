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

import numpy as np
import torch

import scipy as sp

import scipy.ndimage

import torch.fft

from enum import Enum

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
    filtered = [sp.ndimage.filters.gaussian_filter(monochromatic, sigma=sigma)
                        for monochromatic in image.cpu()]
    return torch.Tensor(filtered).to(image.device)

class FilterType(Enum):
    Gaussian=0
    Brickwall=1

class FilteringConfig:
    def __init__(self, filter_type, sigma):
        self.filter_type = filter_type
        self.sigma = sigma

def apply_filter(image, ftr_conf=6):
    if type(ftr_conf) is not FilteringConfig:
        ftr_conf = FilteringConfig(FilterType.Gaussian, ftr_conf)
    return {
       FilterType.Gaussian: lambda img:
          apply_gaussian_filter(img, ftr_conf.sigma)
     , FilterType.Brickwall: lambda img:
          apply_brickwall_filter(img, ftr_conf.sigma)
     }[ftr_conf.filter_type](image)
