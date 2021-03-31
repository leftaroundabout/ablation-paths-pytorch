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


# Algorithm for making a path monotonic. This is a Python port of a
# well-tested reference implementation in Haskell,
# https://github.com/leftaroundabout/monotone-path/blob/master/Data/Path/Monotone.hs

import numba as nb
import numpy as np

@nb.experimental.jitclass([
    ('xMin', nb.int64)
  , ('xMax', nb.int64)
  , ('yMin', nb.float64)
  , ('yMax', nb.float64)
  ])
class IntervalWRange:
    def __init__(self, xMin, xMax, yMin, yMax):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
    def __repr__(self):
        return "IntervalsWRange(%g,%g,%g,%g)" % (
                  self.xMin, self.xMax, self.yMin, self.yMax)

@nb.jit(nb.types.Tuple((nb.int64[:], nb.int64[:], nb.float64[:], nb.float64[:]))(nb.int64))
def alloc_iwrs(n):
    return ( np.zeros(n, dtype=np.int64)
           , np.zeros(n, dtype=np.int64)
           , np.zeros(n, dtype=np.float64)
           , np.zeros(n, dtype=np.float64) )

@nb.jit(nopython=True)
def decreases_here(pth, i):
    return float(pth[i]) > float(pth[i+1])

@nb.jit(nopython=True)
def decreasing_intervals(pth):
    i: int32 = 0
    l = len(pth)
    n_decr_intvs = 0
    while i < l-1:
        if decreases_here(pth, i):
            ir = i+1
            while ir < l-1 and decreases_here(pth, ir):
                ir += 1
            n_decr_intvs += 1
            i = ir
        i += 1
    result = alloc_iwrs(n_decr_intvs)
    i = 0
    j = 0
    while i < l-1:
        if decreases_here(pth, i):
            ir = i
            while ir < l-1 and decreases_here(pth, ir):
                ir += 1
            result[0][j] = i       # xMin
            result[1][j] = ir      # xMax
            result[2][j] = pth[ir] # yMin
            result[3][j] = pth[i]  # yMax
            i = ir
            j += 1
        i += 1
    return result

@nb.jit(nopython=True)
def grow_decreasing_intv(pth, illim, irlim, intv):
    ym = (intv.yMin + intv.yMax)/2
    def l_bound():
        i = intv.xMin
        while i>illim and pth[i-1] > ym:
            i -= 1
        return i
    def r_bound():
        i = intv.xMax
        while i < irlim and pth[i+1] < ym:
            i += 1
        return i
    return IntervalWRange(l_bound(), r_bound(), intv.yMin, intv.yMax)

def break_on_cond(cond, l):
    i=0
    le = len(l)
    while i < le and not cond(l[i]):
        i += 1
    return (l[:i], l[i:])

def merge_overlapping_intvs(grown, orig):
    def go(grown_remaining, orig_remaining):
        if len(grown_remaining)==0:
            return grown_remaining
        this_intv = grown_remaining[0]
        overlapping, rest = break_on_cond(
              lambda iv: iv[0].xMin > this_intv.xMax+1
                      or iv[0].yMin + iv[0].yMin > this_intv.yMax + this_intv.yMax
            , go(grown_remaining[1:], orig_remaining[1:]) )
        yb = np.amin([this_intv.yMin] + [iv.yMin for iv, _ in overlapping])
        yt = np.amax([this_intv.yMax] + [iv.yMax for iv, _ in overlapping])
        xr = np.amax([this_intv.xMax] + [iv.xMax for iv, _ in overlapping])
        return [( IntervalWRange(this_intv.xMin, xr, yb, yt)
                , [this_intv] + [s for _, si in overlapping for s in si])
               ] + rest
    return go(grown, orig)

def project_monotone_lInftymin(pth):
    def grow_and_merge(ivs):
        xMins, xMaxs, yMins, yMaxs = ivs
        while True:
            n = len(xMins)
            if n==0:
                return []
            grown = [None] * n
            m = IntervalWRange(xMins[0], xMaxs[0], yMins[0], yMaxs[0])
            if n>1:
                grown[0] = grow_decreasing_intv(pth, 0, xMins[1], m)
                for i in range(n-2):
                    lb = xMaxs[i]
                    rb = xMins[i+2]
                    m = IntervalWRange(xMins[i+1], xMaxs[i+1], yMins[i+1], yMaxs[i+1])
                    grown[i+1] = grow_decreasing_intv(pth, lb,rb, m)
                m = IntervalWRange(xMins[n-1], xMaxs[n-1], yMins[n-1], yMaxs[n-1])
                grown[n-1] = grow_decreasing_intv(pth, xMaxs[n-2], 1, m)
            else:
                grown[0] = grow_decreasing_intv(pth, 0, 1, m)

            merged = merge_overlapping_intvs(
                         grown
                       , list([IntervalWRange(xMins[i], xMaxs[i], yMins[i], yMaxs[i])
                                 for i in range(n)])
                       )
            if all([len(src) <= 1 for _, src in merged]):
                return [iv for iv, _ in merged]
        
            xMins = np.array([np.amin([iv.xMin for iv in subIvs]) for _, subIvs in merged ])
            xMaxs = np.array([np.amax([iv.xMax for iv in subIvs]) for _, subIvs in merged ])
            yMins = np.array([np.amin([iv.yMin for iv in subIvs]) for _, subIvs in merged ])
            yMaxs = np.array([np.amax([iv.yMax for iv in subIvs]) for _, subIvs in merged ])

    for iv in grow_and_merge(decreasing_intervals(pth)):
        ym = (iv.yMin + iv.yMax)/2
        pth[iv.xMin : iv.xMax+1] = np.array([ym] * (iv.xMax+1 - iv.xMin))
