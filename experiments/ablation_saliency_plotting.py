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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# from monotone_paths import project_monotone_lInftymin
# from ablation import compute_square_intensity
from ablation_paths import masked_interpolation, find_class_transition, resample_to_reso

def mpplot_ablpath_score( model, x, baseline, abl_seqs, label_nr=None
                        , tgt_subplots=None, savename=None
                        , extras={}
                        , pretty_method_names={} ):
    abl_series = abl_seqs.items()
    fig, axs = ( plt.subplots(len(abl_series))
               ) if tgt_subplots is None else (None, tgt_subplots)
    if tgt_subplots is not None:
        assert(len(tgt_subplots)==len(abl_series) and savename is None)
    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))

    all_predictions = {method: torch.softmax(model(torch.stack(
                               masked_interpolation(x, baseline, abl_seq))
                          ), dim=1).detach()[:,label_nr]
                       for method, abl_seq in abl_series }

    def as_domain(method):
        n_spl = all_predictions[method].shape[0]
        return np.linspace(1/(n_spl+1), 1-1/(n_spl+1), n_spl)
    
    for i, (method, abl_seq) in enumerate(abl_series):
        predictions = all_predictions[method]
        sf = axs[i] if len(abl_series)>1 else axs
        sf.fill_between(as_domain(method), predictions.cpu().numpy()
                       , alpha=0.25)
        for method_c, _ in abl_series:
            sf.plot( as_domain(method_c)
                   , all_predictions[method_c].cpu().numpy()
                   , color= (0,0.3,0.5,1) if method_c==method
                            else (0.5,0.5,0,0.5) )
        if method in extras:
            extras[method](sf)
        sf.text(0.1, 0.1, "score %.3g" % torch.mean(predictions))
        sf.set_title(pretty_method_names[method] if method in pretty_method_names
                       else method)
        sf.xaxis.set_major_formatter(mtick.FuncFormatter(
                 lambda x, _: '{:.0%}'.format(x) ))
        if i==len(abl_series)-1:
            sf.set_xlabel('Ablation')
    if savename:
        savepath = 'images/ablation/score-compare/'+savename+'.pdf'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath)


def retrv(t):
    return t.cpu().detach().numpy()

def mp_show_image(sp, im):
    sp.set_aspect('equal')
    sp.axis('off')
    sp.imshow(retrv(im.transpose(0,2) + 1)/2)


def show_mask_combo_at_classTransition(model, x, baseline, abl_seq, tgt_subplots=None, **kwargs):
    nCh, w, h = x.shape
    def apply_mask(y, mask):
        return y * resample_to_reso(mask.unsqueeze(0), (w,h)).repeat(nCh,1,1)
    transition_loc = find_class_transition(model, x, baseline, abl_seq, **kwargs)
    mask = abl_seq[transition_loc]
    x_masked = apply_mask(x, 1 - mask)
    bl_masked = apply_mask(baseline, mask)
    fig,axs = ( plt.subplots(1,3)
              ) if tgt_subplots is None else (None,tgt_subplots)
    if tgt_subplots is not None:
        assert (len(tgt_subplots)==3)
    for i,im in enumerate([x_masked, x_masked+bl_masked, bl_masked]):
        mp_show_image(axs[i], im)
    return transition_loc


