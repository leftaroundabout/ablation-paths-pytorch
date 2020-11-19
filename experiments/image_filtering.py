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

def apply_filter(image, sigma=6):
    filtered = [sp.ndimage.filters.gaussian_filter(monochromatic, sigma=sigma)
                        for monochromatic in image.cpu()]
    return torch.Tensor(filtered).to(image.device)

