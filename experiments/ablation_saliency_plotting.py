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
import torchvision
from imageclassifier_model import TrainedTimmModel
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid
from matplotlib.colors import to_rgb

import panel as pn
import holoviews as hv
import pandas as pd
import hvplot.pandas

# from monotone_paths import project_monotone_lInftymin
# from ablation import compute_square_intensity
from ablation_paths import ( masked_interpolation
                           , find_class_transition
                           , resample_to_reso
                           , AblPathObjective )
from image_filtering import apply_gaussian_filter

def mpplot_ablpath_score( model, x, baselines, abl_seqs, label_nr=None
                        , tgt_subplots=None, savename=None
                        , label_name=None, labels=None
                        , extras={}
                        , pretty_method_names={}
                        , classification_name=None
                        , include_endpoints=True
                        , objective=AblPathObjective.FwdLongestRetaining ):
    if callable(baselines):
        baseline_samples = [ baselines()
                              for i in range(12) ]
    else:
        baseline_samples = baselines
    if classification_name is None:
        if type(model) is TrainedTimmModel:
            classification_name = model.timm_model_name
    abl_series = abl_seqs.items()
    fig, axs = ( plt.subplots(len(abl_series), squeeze=False)
               ) if tgt_subplots is None else (None, tgt_subplots)
    if tgt_subplots is not None:
        if (len(tgt_subplots)!=len(abl_series)):
            raise ValueError("Need %i subplot rows, got %i." %(len(abl_series), len(tgt_subplots)))
        assert(savename is None)
    classif_top_label = torch.argmax(model(x.unsqueeze(0)))
    if label_nr is None:
        label_nr = classif_top_label

    def relevant_predictions():
        complete_classif, contrast_classif = [
                   {method: [ torch.softmax(model(torch.stack(
                                 masked_interpolation( x, baseline, co(abl_seq)
                                                     , include_endpoints=include_endpoints ))
                                            ), dim=1).detach()
                                     for baseline in baseline_samples ]
                              for method, abl_seq in abl_series }
                 for co in [lambda abls: abls, lambda abls: 1-abls] ]
        return {**{ vis_class:
                  { method: [ cl[:,label_acc] for cl in cls ]
                   for method, cls in complete_classif.items() }
                for vis_class, label_acc in [('focused', label_nr)
                                            ,('top', classif_top_label)]
                 if classif_top_label!=label_nr or vis_class=='focused' }
               , 'contrast': { method: [ cl[:,label_nr] for cl in cls ]
                   for method, cls in contrast_classif.items() }
               }
    all_predictions = relevant_predictions()

    def as_domain(method, sampleid):
        n_spl = all_predictions['focused'][method][sampleid].shape[0]
        return ( np.linspace(0, 1, n_spl)
                if include_endpoints
                 else np.linspace(1/(n_spl+1), 1-1/(n_spl+1), n_spl) )
    
    predictions_stats = {vis_class: {
        method:
         { fdescr: np.array([f([float(pred[j]) for pred in predictions])
                           for j in range(0, predictions[0].shape[0])])
            for fdescr, f in (
                  [('median', np.median), ('min', np.min), ('max', np.max)]
                   if vis_class in ['focused', 'contrast']
                          else [('median', np.median)] )
         }
         for method, predictions in rel_predicts.items() }
      for vis_class, rel_predicts in all_predictions.items() }

    for i, (method, abl_seq) in enumerate(abl_series):
        predictions = all_predictions['focused'][method]
        contrasts = all_predictions['contrast'][method]
        sf = axs[i] if tgt_subplots is not None else axs[i][0]
        for stat_d in ['median', 'min', 'max']:
            sf.fill_between( as_domain(method,0)
                           , predictions_stats['contrast'][method][stat_d]
                            if objective is AblPathObjective.BwdQuickestDissipating
                            else predictions_stats['focused'][method][stat_d]
                           , predictions_stats['contrast'][method][stat_d]
                            if objective is AblPathObjective.FwdRetaining_BwdDissipating
                            else 0
                           , alpha=0.1
                           , color = (0,0.3,0.5,1) )
        for method_c, _ in abl_series:
            sf.plot( as_domain(method_c,0)
                   , predictions_stats['focused'][method_c]['median']
                   , color= (0,0.3,0.5,1) if method_c==method
                            else (0.5,0.5,0,0.5) )
        if classif_top_label!=label_nr:
            sf.plot( as_domain(method,0)
                   , predictions_stats['top'][method]['median']
                   , color= (1.0,0.1,0,0.1) )
        if method in extras:
            extras[method](sf)
        def trapez(t):
            return torch.mean(torch.cat([(t[0:1]+t[-1:])/2, t[1:-1]]))
        goodnesses = [ predictions[j] if objective is AblPathObjective.FwdLongestRetaining
                       else contrasts[j] if objective is AblPathObjective.BwdQuickestDissipating
                       else predictions[j]-contrasts[j]
                      for j in range(len(predictions)) ]
        this_score = ( torch.mean(torch.stack([trapez(g) for g in goodnesses]))
                                          if include_endpoints
                                          else torch.mean(torch.stack(goodnesses)) )
        if label_name is None and labels is not None:
            label_name = labels[label_nr]
        score_descr = ( "%s → %s" % (classification_name, label_name)
                         if classification_name is not None and label_name is not None
                       else classification_name if classification_name is not None
                       else label_name          if label_name is not None
                       else None )
        sf.text(0.1, 0.1, "score %.3g%s"
                               % ( this_score, " (%s)" % score_descr
                                               if score_descr is not None else "" )
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

class MaskOverlayed:
    def __init__(self, img_topic, mask_overlayer):
        self.img_topic = img_topic
        self.mask_overlayer = mask_overlayer

class AverageMaskOverlayed:
    def __init__(self, img_topic, mask_overlayer):
        self.img_topic = img_topic
        self.mask_overlayer = mask_overlayer

class MaskDisplaying:
    def __init__(self, colourmap='hot'):
        self.colourmap = colourmap

def overlay_mask_contours(y, mask, contour_colour, contour_width=2):
    nCh, w, h = y.shape
    msk_cheby_threshold = (float(torch.max(mask)) + float(torch.min(mask)))/2
    crossings_horiz = torch.abs(torch.diff(torch.sign(
                                  resample_to_reso(mask.unsqueeze(0), (w+1,h)) # strictly speaking we should apply
                                         - msk_cheby_threshold)                # padding instead of rescaling to w+1
                        , axis=1))
    crossings_vert = torch.abs(torch.diff(torch.sign(
                                  resample_to_reso(mask.unsqueeze(0), (w,h+1))
                                         - msk_cheby_threshold)
                        , axis=2))
    contour = torch.zeros_like(crossings_horiz)
    contour[crossings_horiz + crossings_vert > 0] = 1
    if contour_width > 1:
        contour = apply_gaussian_filter(contour, contour_width)
        contour[contour < 1/(4*contour_width)] = 0
        contour[contour > 0] = 1
    overlayed = y.clone()
    if contour_colour.shape==(nCh,):
        for i in range(nCh):
            overlayed[i, contour[0] > 0
                     ] = contour_colour[i]
    return overlayed

def toHSV(y):
    return torchvision.transforms.ToTensor()(
                 torchvision.transforms.ToPILImage()((y + 1)/2).convert("HSV")
            ).to(y.device)

def fromHSV(y):
    return 2*torchvision.transforms.ToTensor()(
                 torchvision.transforms.ToPILImage(mode='HSV')(y)
               .convert("RGB")).to(y.device) - 1

def desaturate(y, saturation=0.3):
    hsvI = toHSV(y)
    hsvI[1,:,:] *= saturation
    return fromHSV(hsvI)

def overlay_mask_as_hue(y, mask, y_saturation=0.3, y_prominence=1.0):
    nCh, w, h = y.shape

    # 0.7 is the hue of blue, 0 of red.
    mask_hue = 0.7 * resample_to_reso(mask.unsqueeze(0), (w,h))
    
    spectral_mask_hsv = torch.ones_like(y)
    spectral_mask_hsv[0,:,:] = mask_hue
    spectral_mask = fromHSV(spectral_mask_hsv)

    return (3*y_prominence*desaturate(y, y_saturation) + spectral_mask
             )/(3*y_prominence+1)

def overlay_mask_deemphasizeirrelevant(
                y, mask
              , outside_saturation=0.3, outside_contrast=0.5, outside_brightness=-0.3 ):
    if not 0<=outside_contrast<=1:
        raise ValueError("Contrast must be between 0 and 1")
    if np.abs(outside_brightness) > 1 - outside_contrast:
        raise ValueError("Brightness adjustment must be between {} and {}"
                           .format(outside_contrast-1, 1-outside_contrast))
    return ( masked_interpolation
              ( y, outside_contrast*desaturate(y, outside_saturation)
                     + outside_brightness
              , mask.unsqueeze(0) )[0] )

def overlay_mask_argmin(
                y, mask
              , crosshair_radius=5
              , crosshair_colour='red' ):
    crosshair_colour = to_rgb(crosshair_colour)
    iys, ixs = torch.meshgrid([torch.arange(n) for n in y.shape[1:]])
    iys_mask, ixs_mask = torch.meshgrid([
          torch.tensor(np.linspace(0, y.shape[j+1], mask.shape[j], endpoint=False))
        for j in [0,1] ])
    iflat_am = torch.argmin(mask)
    iy_am = round(float(iys_mask.flatten()[iflat_am]))
    ix_am = round(float(ixs_mask.flatten()[iflat_am]))
    ch_hori = (ixs==ix_am) & (iys >= iy_am-crosshair_radius
                         ) & (iys <= iy_am+crosshair_radius)
    ch_vert = (iys==iy_am) & (ixs >= ix_am-crosshair_radius
                         ) & (ixs <= ix_am+crosshair_radius)
    for j in range(y.shape[0]):
        y[j][ch_hori | ch_vert] = crosshair_colour[j]*2 - 1
    
    return y

default_mask_combo_img_views = ['target_masked', 'interpolation_result', 'baseline_antimasked']

def mpplotgrid_score_below_image( n_abl_seqs, n_imgviews, n_columns=1, n_extra_rows=0
                                , figsize=None, gridspec_kw={} ):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    n_abl_seqs_per_column = -(n_abl_seqs//-n_columns)
    gs = grid.GridSpec( 2*n_abl_seqs_per_column + n_extra_rows
                      , n_imgviews*n_columns, **gridspec_kw )
    axs = np.asarray(
           [ ( [ fig.add_subplot(gs[ 2*(j // n_columns) + 1
                                   , (j % n_columns)*n_imgviews
                                      : (j % n_columns + 1)*n_imgviews ]) ]
              + [fig.add_subplot(gs[ 2*(j // n_columns)
                                   , (j % n_columns)*n_imgviews + i ])
                                          for i in range(n_imgviews) ] )
            for j in range(n_abl_seqs) ] )
    return fig, axs

def mpplotgrid_for_maskcombos( n_abl_seqs, n_imgviews, n_columns=1
                             , n_extra_rows=0, extra_row_height=2
                             , scoreplot_below_images=True ):
    n_abl_seqs_per_column = -(n_abl_seqs//-n_columns)  # rounding towards +∞
    if scoreplot_below_images:
        fig,axs = mpplotgrid_score_below_image(
                               n_abl_seqs, n_imgviews, n_columns=n_columns
                             , n_extra_rows=n_extra_rows
                             , figsize=( 3*n_imgviews*n_columns
                                       , 4.5*n_abl_seqs_per_column + extra_row_height*n_extra_rows )
                             , gridspec_kw={'height_ratios':
                                            [h for _ in range(n_abl_seqs_per_column)
                                                for h in [3,1]]
                                                              + [2 for _ in range(n_extra_rows)] } )
    else:
        n_sidecells = n_imgviews+1
        fig,axsg = plt.subplots( n_abl_seqs_per_column, n_sidecells*n_columns, squeeze=False
                               , figsize=((4+2*n_imgviews)*n_columns, 2*(n_abl_seqs+n_extra_rows))
                               , gridspec_kw={'width_ratios': ([2.5] + [1 for _ in range (n_imgviews)])
                                                               * n_columns } )
        axs = np.asarray([ [axsg[j//n_columns, (j%n_columns)*n_sidecells + x]
                            for x in range(n_sidecells)]
                          for j in range(n_abl_seqs) ])
    return fig, axs

def show_mask_combo_at_classTransition( model, x, baseline, abl_seq
                                      , tgt_subplots=None, scoreplot_below_images=True
                                      , manual_loc_select=None
                                      , img_views=default_mask_combo_img_views
                                      , **kwargs
                                      ):
    nCh, w, h = x.shape
    def apply_mask(y, mask):
        return y * resample_to_reso(mask.unsqueeze(0), (w,h)).repeat(nCh,1,1)
    transition_loc = (find_class_transition(model, x, baseline, abl_seq, **kwargs) - 1
        ) if manual_loc_select is None else (
         round(manual_loc_select * (abl_seq.shape[0]+2)) - 1 )
    mask = ( torch.zeros_like(abl_seq[0]) if transition_loc<0
         else abl_seq[transition_loc] if transition_loc<abl_seq.shape[0]
         else torch.ones_like(abl_seq[0]) )
    x_masked = apply_mask(x, 1 - mask)
    bl_masked = apply_mask(baseline, mask)
    if tgt_subplots is None:
        fig,axs = mpplotgrid_for_maskcombos(n_abl_seqs, n_imgviews, scoreplot_below_images)
        axs = axs[0]
    else:
        assert (len(tgt_subplots)==len(img_views))
        fig,axs = (None, tgt_subplots)
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
        elif isinstance(imview, AverageMaskOverlayed):
            avg_mask = torch.mean(abl_seq, dim=(0,))
            im = imview.mask_overlayer(view_options[imview.img_topic], avg_mask)
            mp_show_image(axs[i], im)
        elif isinstance(imview, MaskOverlayed):
            im = imview.mask_overlayer(view_options[imview.img_topic], mask)
            mp_show_image(axs[i], im)
        elif isinstance(imview, MaskDisplaying):
            mp_show_mask(axs[i], 1 - mask, imview.colourmap)
        else:
            raise TypeError("Unknown image view type "+str(imview))
    return transition_loc


def get_dataframe(results, labels, namer=lambda i: 'im{}'.format(i)):
    pd_res = (
        pd.concat([
            pd.DataFrame(labels.items(), columns=['ind', 'label']).drop(columns=['ind']), 
            pd.DataFrame( torch.nn.Softmax(dim=1)(results).cpu().detach().numpy().T
                        , columns=[namer(i) for i in range(len(results))] )
                  ],
                  axis=1)
        .set_index('label')
    )
    return pd_res
def show_histogram( df, name='im0', width=400, height=400
                  , columns_shown=6, guaranteed_labels=[], **kwargs ):
    if len(guaranteed_labels)>columns_shown:
        raise ValueError("Cannot show requested labels %s in only %i columns"
                            % (guaranteed_labels, columns_shown))
    score_select = df.sort_values(by=name, ascending=False).iloc[:columns_shown]
    preguaranteed = score_select.index.map(lambda lbl: lbl in guaranteed_labels)
    n_preguaranteed = len(score_select[preguaranteed])
    n_non_preguaranteed = columns_shown + n_preguaranteed - len(guaranteed_labels)
    non_preguaranteed = df.sort_values(by=name, ascending=False).iloc[:n_non_preguaranteed]
    guaranteed = df.loc[guaranteed_labels]
    score_select = non_preguaranteed.merge(guaranteed, how='outer', on=['label', name])
    
    score_select.index = score_select.index.map(lambda l: '─▶ '+l if l in guaranteed_labels else l)
    
    def label_fmt(label):
      # if label in guaranteed_labels:
      #     l = '─▶ '+l
        if len(label)>11:
           label = label[0:11]+"…"
        return label

    return ( score_select.reset_index()
               .hvplot.bar( x='label', y=name).opts(
                      hv.opts.Bars( xrotation=60
                                  , xformatter=label_fmt
                                  , width=width, height=height, ylim=(0,1) )
                    , **kwargs ) )
def classification_histogram( model, x, labels, name='im0', **kwargs ):
    return show_histogram(get_dataframe(model(x), labels=labels), **kwargs)

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

class MaskViewOverlay:
    def __init__(self, precomposes=[]):
        self.precomposes=precomposes
    def __call__(self, y, mask):
        for f in reversed(self.precomposes):
            y = f(y, mask)
        return y
    def __mul__(self, other):
        return MaskViewOverlay([self, other])

class DeemphasizeIrrelevant(MaskViewOverlay):
    def __init__(self, outside_saturation=0.5, outside_contrast=2/3, outside_brightness=-0.2):
        self.outside_saturation=outside_saturation
        self.outside_contrast=outside_contrast
        self.outside_brightness=outside_brightness
    def __call__(self, y, mask):
        return overlay_mask_deemphasizeirrelevant( y, mask
                 , outside_saturation=self.outside_saturation
                 , outside_contrast=self.outside_contrast
                 , outside_brightness=self.outside_brightness )

class MaskAsHueOverlay(MaskViewOverlay):
    def __init__(self, y_saturation=0.3, y_prominence=1.0):
        self.y_saturation=y_saturation
        self.y_prominence=y_prominence
    def __call__(self, y, mask):
        return overlay_mask_as_hue( y, mask
                 , y_saturation=self.y_saturation
                 , y_prominence=self.y_prominence )

class FixedSaliencyOverlay(MaskViewOverlay):
    def __init__(self, saliency, y_saturation=0.3, y_prominence=1.0, normalize_saliency_range=True):
        if normalize_saliency_range:
            saliency -= torch.min(saliency)
            saliency /= torch.max(saliency)
        self.saliency = saliency
        self.y_saturation=y_saturation
        self.y_prominence=y_prominence
    def __call__(self, y, mask):
        return overlay_mask_as_hue( y, self.saliency
                 , y_saturation=self.y_saturation
                 , y_prominence=self.y_prominence )

class MaskMidlevelContour(MaskViewOverlay):
    def __init__(self, contour_colour=torch.tensor([1,-1,-0.5]), contour_width=2):
        self.contour_colour=contour_colour
        self.contour_width=contour_width
    def __call__(self, y, mask):
        return overlay_mask_contours( y, mask
           , contour_colour=self.contour_colour
           , contour_width=self.contour_width )

class MaskArgminOverlay(MaskViewOverlay):
    def __init__(self, crosshair_radius=5, crosshair_colour='red' ):
        self.crosshair_radius=crosshair_radius
        self.crosshair_colour=crosshair_colour
    def __call__(self, y, mask):
        return overlay_mask_argmin(
                y, mask
              , crosshair_radius=self.crosshair_radius
              , crosshair_colour=self.crosshair_colour )

class OverlayWithFullySaturatedMask(MaskViewOverlay):
    def __init__(self, overlay_method):
        self.overlay_method = overlay_method
    def __call__(self, y, mask):
        nCh, w, h = y.shape
        msk_cheby_threshold = (float(torch.max(mask)) + float(torch.min(mask)))/2
        mask_hr = resample_to_reso(mask.unsqueeze(0), (w,h))[0]
        if msk_cheby_threshold<0.5:
            mask_hr[mask_hr <= msk_cheby_threshold] = 0
            mask_hr[mask_hr > msk_cheby_threshold] = 1
        else:
            mask_hr[mask_hr < msk_cheby_threshold] = 0
            mask_hr[mask_hr >= msk_cheby_threshold] = 1
        return self.overlay_method(y, mask_hr)

class OverlayWithManuallyOverridenMask(MaskViewOverlay):
    def __init__(self, overlay_method, mask_override):
        self.overlay_method = overlay_method
        self.mask_override = mask_override
    def __call__(self, y, mask):
        return self.overlay_method(y, self.mask_override)

class ManualOverrideMaskOverlayed(MaskOverlayed):
    def __init__(self, img_topic, mask_overlayer, mask_override):
        self.img_topic = img_topic
        self.mask_overlayer = OverlayWithManuallyOverridenMask(mask_overlayer, mask_override)


intuitive_saliency_mask_combo_img_views = [
    MaskOverlayed( 'target_original'
                 , MaskMidlevelContour()
                  * OverlayWithFullySaturatedMask(DeemphasizeIrrelevant()) )
  , MaskOverlayed( 'interpolation_result', MaskMidlevelContour() )
  ]

def interactive_view_mask( abl_seq, x=None, baseline=None, model=None, labels=None
                         , view_x=DeemphasizeIrrelevant()
                         , view_masks=False, view_interpolation=auto, view_classification=auto
                         , view_scoregraph=False
                         , add_overlay_mask_contours=False
                         , add_overlay_mask_as_hue=False
                         , image_valrange=(-1,1)
                         , focused_labels=[]
                         , objective_class=None
                         , viewers_size=auto, classification_name=None, **kwargs ):
    n_ablseq = abl_seq.shape[0]
    def to_unit_range(y):
        return (2*y - image_valrange[0] - image_valrange[1]
               ) / (image_valrange[1] - image_valrange[0])
    torchdevice = x.device if x is not None else None
    if classification_name is None:
        if type(model) is TrainedTimmModel:
            classification_name = model.timm_model_name
    inter_select = pn.widgets.IntSlider(start=0, end=n_ablseq+1)
    complement_select = pn.widgets.Checkbox(name='Complement masks')
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
    abl_seq_wEndpoints = torch.cat(
                          [ ooc(torch.cat( [ torch.zeros_like(abl_seq[0:1])
                                           , abl_seq
                                           , torch.ones_like(abl_seq[0:1]) ] ))
                           for ooc in [lambda abls: abls, lambda abls: 1-abls] ] )
    n_tested = abl_seq_wEndpoints.shape[0]
    interpol_seq = masked_interpolation(x, baseline, abl_seq_wEndpoints
         ) if view_interpolation or view_classification else None
    interpol_seq_normrng = [to_unit_range(i) for i in interpol_seq]
    if add_overlay_mask_as_hue:
        interpol_seq_maskhint = [ overlay_mask_as_hue
                                           ( interpol_seq_normrng[i], abl_seq_wEndpoints[i] )
                                   for i in range(n_tested) ]
    else:
        interpol_seq_maskhint = interpol_seq_normrng
    if view_x is True:
        x_view_seq = [ x for i in range(n_tested) ]
        do_x_view = True
    elif isinstance(view_x, MaskViewOverlay):
        x_view_seq = [ view_x( to_unit_range(x), abl_seq_wEndpoints[i] )
                                   for i in range(n_tested) ]
        do_x_view = True
    else:
        do_x_view = False
    if add_overlay_mask_contours:
        interpol_seq_contours = [ overlay_mask_contours
                                           ( interpol_seq_maskhint[i], abl_seq_wEndpoints[i]
                                           , contour_colour=torch.tensor([1,-1,-0.5]) )
                                   for i in range(n_tested) ]
        interpol_seq_maskhint = interpol_seq_contours
    if view_classification or view_scoregraph:
        classifications = model(torch.stack(interpol_seq).to(torchdevice)).detach().clone()
        clshape = classifications.shape
        classifications = classifications.reshape(n_tested, clshape[1])
    if view_scoregraph:
        n_scoregraph = 128
        masses_od = np.array([float(torch.mean(abl_seq_wEndpoints[i]))
                               for i in range(n_ablseq+2)])
        masses = np.linspace(0,1,n_scoregraph)
        top_class = int(torch.argmax(classifications[0]))
        if objective_class is None:
            objective_class = top_class
        objective_class_probs_all = torch.softmax(classifications, dim=1
                                      )[:,objective_class].cpu().numpy()
        objective_class_probs_opt = np.interp(masses, masses_od
                  , objective_class_probs_all[:n_ablseq+2] )
        objective_class_probs_pess = np.interp(masses, masses_od
                  , objective_class_probs_all[n_ablseq+2:] )
    def hvRGB(y):
        return hv.RGB((y.transpose(1,2).transpose(0,2).cpu().numpy() + 1)/2
                      ).opts(**hvopts_img)
    def show_intermediate( complement, i
                         , enable_scoregraph=True, enable_others=True ):
        c = n_ablseq+2 if complement else 0
        intensity = abl_seq_wEndpoints[c+i].cpu().numpy()
        views = []
        if do_x_view:
            xview_img = x_view_seq[c+i]
            xview = hvRGB(xview_img)
            views = views + [xview]
        if view_masks and enable_others:
            maskview = hv.Image(intensity).opts(**hvopts).redim.range(z=(1,0))
            views = views + [maskview]
        if view_interpolation and enable_others:
            interpol_img = interpol_seq_maskhint[c+i]
            interpolview = hvRGB(interpol_img)
            views = views + [interpolview]
        if view_classification and enable_others:
            dfopts = {} if classification_name is None else {
                             'namer': lambda _: classification_name}
            classifview = show_histogram( get_dataframe(classifications[c+i : c+i+1], labels, **dfopts)
                                           , guaranteed_labels=focused_labels, **hvopts_classif )
            views = views + [classifview]
        if view_scoregraph and enable_scoregraph:
            sgv_opts = { 'width': viewers_size*sum([view_masks, view_interpolation, view_classification])
                       , 'height': viewers_size//2
                       , 'xlim':(0,1), 'ylim':(0,1)
                       , 'xlabel':'t', 'ylabel':'classif←%s'%labels[objective_class]
                       , 'axiswise':True }
            scoregraphview = ( hv.Area(( masses
                                       , objective_class_probs_opt
                                       , np.minimum(
                                            objective_class_probs_opt
                                          , objective_class_probs_pess ) )
                                      , vdims=['yopt', 'ypess'] )
                                .opts(fill_alpha=0.3, **sgv_opts)
                             * hv.Curve((masses
                                          , objective_class_probs_pess
                                           if complement
                                           else objective_class_probs_opt))
                                .opts(color='blue', **sgv_opts)
                             * hv.Curve(([masses_od[i],masses_od[i]], [0,1]))
                                .opts(color='black', **sgv_opts)
                             ).opts(**sgv_opts)
            views = views + [scoregraphview]

        return views
    if view_scoregraph:
        return pn.Column( pn.Row(complement_select, inter_select)
                        , pn.depends(complement_select.param.value, inter_select.param.value)
                                    (lambda cp,i: hv.Layout(show_intermediate(
                                                      cp, i, enable_scoregraph=False)))
                        , pn.depends(complement_select.param.value, inter_select.param.value)
                                    (lambda cp,i: hv.Layout(show_intermediate(
                                                      cp, i, enable_others=False)))
                        )
    else:
        return pn.Column( inter_select
                        , pn.depends(inter_select.param.value)
                                    (lambda i: hv.Layout(show_intermediate(i))) )

