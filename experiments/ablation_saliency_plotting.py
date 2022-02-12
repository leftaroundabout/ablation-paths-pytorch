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

import panel as pn
import holoviews as hv
import pandas as pd
import hvplot.pandas

# from monotone_paths import project_monotone_lInftymin
# from ablation import compute_square_intensity
from ablation_paths import masked_interpolation, find_class_transition, resample_to_reso

def mpplot_ablpath_score( model, x, baselines, abl_seqs, label_nr=None
                        , tgt_subplots=None, savename=None
                        , extras={}
                        , pretty_method_names={}
                        , include_endpoints=True ):
    if callable(baselines):
        baseline_samples = [ baselines()
                              for i in range(12) ]
    else:
        baselines_samples = [baselines]
    abl_series = abl_seqs.items()
    fig, axs = ( plt.subplots(len(abl_series))
               ) if tgt_subplots is None else (None, tgt_subplots)
    if tgt_subplots is not None:
        if (len(tgt_subplots)!=len(abl_series)):
            raise ValueError("Need %i subplot rows, got %i." %(len(abl_series), len(tgt_subplots)))
        assert(savename is None)
    if label_nr is None:
        label_nr = torch.argmax(model(x.unsqueeze(0)))

    all_predictions = {
         method: [ torch.softmax(model(torch.stack(
                               masked_interpolation( x, baseline, abl_seq
                                                   , include_endpoints=include_endpoints ))
                          ), dim=1).detach()[:,label_nr]
                    for baseline in baseline_samples ]
            for method, abl_seq in abl_series }

    def as_domain(method, sampleid):
        n_spl = all_predictions[method][sampleid].shape[0]
        return ( np.linspace(0, 1, n_spl)
                if include_endpoints
                 else np.linspace(1/(n_spl+1), 1-1/(n_spl+1), n_spl) )
    
    predictions_stats = {
        method:
         { fdescr: np.array([f([float(pred[j]) for pred in predictions])
                           for j in range(0, predictions[0].shape[0])])
            for fdescr, f in [('median', np.median), ('min', np.min), ('max', np.max)]
         }
         for method, predictions in all_predictions.items() }

    for i, (method, abl_seq) in enumerate(abl_series):
        predictions = all_predictions[method]
        sf = axs[i] if len(abl_series)>1 or tgt_subplots is not None else axs
        for stat_d in ['median', 'min', 'max']:
            sf.fill_between( as_domain(method,0)
                           , predictions_stats[method][stat_d]
                           , alpha=0.1
                           , color = (0,0.3,0.5,1) )
        for method_c, _ in abl_series:
            sf.plot( as_domain(method_c,0)
                   , predictions_stats[method_c]['median']
                   , color= (0,0.3,0.5,1) if method_c==method
                            else (0.5,0.5,0,0.5) )
        if method in extras:
            extras[method](sf)
        def trapez(t):
            return torch.mean(torch.cat([(t[0:1]+t[-1:])/2, t[1:-1]]))
        sf.text(0.1, 0.1, "score %.3g" % ( torch.mean(torch.stack([trapez(p) for p in predictions]))
                                          if include_endpoints
                                          else torch.mean(torch.stack(predictions)) )
               )
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

def mp_show_mask(sp, msk, colourmap):
    sp.set_aspect('equal')
    sp.axis('off')
    sp.pcolormesh(retrv(msk.flip(0)), cmap=colourmap)

def mp_show_image(sp, im):
    sp.set_aspect('equal')
    sp.axis('off')
    sp.imshow(retrv(im.transpose(1,2).transpose(0,2) + 1)/2)

default_mask_combo_img_views = ['target_masked', 'interpolation_result', 'baseline_antimasked']

class MaskDisplaying:
    def __init__(self, colourmap='hot'):
        self.colourmap = colourmap

def show_mask_combo_at_classTransition( model, x, baseline, abl_seq, tgt_subplots=None
                                      , manual_loc_select=None
                                      , img_views=default_mask_combo_img_views
                                      , **kwargs
                                      ):
    nCh, w, h = x.shape
    def apply_mask(y, mask):
        return y * resample_to_reso(mask.unsqueeze(0), (w,h)).repeat(nCh,1,1)
    transition_loc = find_class_transition(model, x, baseline, abl_seq, **kwargs
        ) if manual_loc_select is None else (
         int(manual_loc_select * abl_seq.shape[0]) )
    mask = abl_seq[transition_loc]
    x_masked = apply_mask(x, 1 - mask)
    bl_masked = apply_mask(baseline, mask)
    fig,axs = ( plt.subplots(1,len(img_views))
              ) if tgt_subplots is None else (None,tgt_subplots)
    if tgt_subplots is not None:
        assert (len(tgt_subplots)==len(img_views))
    view_options = {
       'target_original': x
     , 'target_masked': x_masked
     , 'interpolation_result': x_masked+bl_masked
     , 'baseline_antimasked': bl_masked
     , 'baseline_original': baseline
     }
    for i,imview in enumerate(img_views):
        if isinstance(imview, str):
            im = view_options[imview]
            mp_show_image(axs[i], im)
        else:
            mp_show_mask(axs[i], 1 - mask, imview.colourmap)
    return transition_loc


def get_dataframe(results, labels, namer=lambda i: 'im{}'.format(i)):
    labels_short = [(i, label if len(label)<12 else label[0:11]+"…")
                     for i, label in labels.items()]
    pd_res = (
        pd.concat([
            pd.DataFrame(labels_short, columns=['ind', 'label']).drop(columns=['ind']), 
            pd.DataFrame( torch.nn.Softmax(dim=1)(results).cpu().detach().numpy().T
                        , columns=[namer(i) for i in range(len(results))] )
                  ],
                  axis=1)
        .set_index('label')
    )
    return pd_res
def show_histogram(df, name='im0', width=400, height=400, columns_shown=6, **kwargs):
    return ( df.sort_values(by=name, ascending=False).iloc[:columns_shown].reset_index()
               .hvplot.bar(x='label', y=name).opts(
                   hv.opts.Bars(xrotation=60, width=width, height=height, ylim=(0,1))
                 , **kwargs) )

# Irritatingly, HoloViews' + operator only forms a semigroup, not monoid,
# so concatenating a variable number of plot objects requires this
# nonstandard "summation".
def nesum(l):
    r = l[0]
    for i in range(1, len(l)):
        r = r + l[i]
    return r

class Auto:
    def __init__(self):
        pass
auto = Auto()

def interactive_view_mask( abl_seq, x=None, baseline=None, model=None, labels=None
                         , view_masks=True, view_interpolation=auto, view_classification=auto
                         , view_scoregraph=False
                         , viewers_size=auto, classification_name=None, **kwargs ):
    torchdevice = x.device if x is not None else None
    inter_select = pn.widgets.IntSlider(start=0, end=len(abl_seq)+2)
    if view_interpolation is auto:
        view_interpolation = (x is not None) and (baseline is not None)
    if view_classification is auto:
        view_classification = (model is not None) and (labels is not None
                             ) and (x is not None) and (baseline is not None)
    if viewers_size is auto:
        viewers_size = 350 if view_interpolation else 600
    hvopts_general = { 'width': viewers_size, 'height': viewers_size }
    hvopts_img = dict( [('data_aspect', 1)], **hvopts_general )
    hvopts = dict( [ ('colorbar', True), ('cmap', 'hot') ]
                 , **dict(hvopts_img.items(), **kwargs) )
    hvopts_classif = hvopts_general.copy()
    if classification_name is not None:
        hvopts_classif['name'] = classification_name
    abl_seq_wEndpoints = torch.cat( [ torch.zeros_like(abl_seq[0:1])
                                    , abl_seq
                                    , torch.ones_like(abl_seq[0:1]) ] )
    interpol_seq = masked_interpolation(x, baseline, abl_seq_wEndpoints
         ) if view_interpolation or view_classification else None
    if view_classification or view_scoregraph:
        classifications = model(torch.stack(interpol_seq).to(torchdevice)).detach().clone()
    if view_scoregraph:
        masses = np.array([float(torch.mean(abl_seq_wEndpoints[i])) for i in range(len(abl_seq)+2)])
        top_class = int(torch.argmax(classifications[0]))
        topclass_probs = torch.softmax(classifications, dim=1)[:,top_class].cpu().numpy()
    def show_intermediate(i, enable_scoregraph=True, enable_others=True):
        intensity = abl_seq_wEndpoints[i].cpu().numpy()
        views = []
        if view_masks and enable_others:
            maskview = hv.Image(intensity).opts(**hvopts).redim.range(z=(1,0))
            views = views + [maskview]
        if view_interpolation and enable_others:
            interpol_img = interpol_seq[i]
            interpolview = hv.RGB((interpol_img.transpose(1,2).transpose(0,2).cpu().numpy() + 1)/2
                              ).opts(**hvopts_img)
            views = views + [interpolview]
        if view_classification and enable_others:
            dfopts = {} if classification_name is None else {'namer': lambda _: classification_name}
            classifview = show_histogram( get_dataframe(classifications[i:i+1], labels, **dfopts)
                                           , **hvopts_classif )
            views = views + [classifview]
        if view_scoregraph and enable_scoregraph:
            sgv_opts = { 'width': viewers_size*sum([view_masks, view_interpolation, view_classification])
                       , 'height': viewers_size//2
                       , 'xlim':(0,1), 'ylim':(0,1)
                       , 'xlabel':'t', 'ylabel':'classif←%s'%labels[top_class]
                       , 'axiswise':True }
            scoregraphview = ( hv.Area((masses, topclass_probs))
                                .opts(**sgv_opts)
                             * hv.Curve(([masses[i],masses[i]], [0,1]))
                                .opts(color='black', **sgv_opts)
                             ).opts(**sgv_opts)
            views = views + [scoregraphview]

        return views
    if view_scoregraph:
        return pn.Column( inter_select
                        , pn.depends(inter_select.param.value)
                                    (lambda i: hv.Layout(show_intermediate(i, enable_scoregraph=False)))
                        , pn.depends(inter_select.param.value)
                                    (lambda i: hv.Layout(show_intermediate(i, enable_others=False)))
                        )
    else:
        return pn.Column( inter_select
                        , pn.depends(inter_select.param.value)
                                    (lambda i: hv.Layout(show_intermediate(i))) )

