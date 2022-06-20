"""
Created on 13 June 2022
count C. elegans embryos on a plate (methdds)
@author: wimth
"""
import numpy as np
import pandas as pd
import re
import sys
import os
import copy
from pathlib import Path
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import sys
from pathlib import Path

from skimage import data, img_as_float, dtype_limits,img_as_int
from skimage.filters import gaussian, laplace,frangi,laplace,threshold_otsu
from skimage.segmentation import active_contour, watershed
from skimage.io import imread,imshow
from skimage.morphology import binary_closing, closing, square, skeletonize
from tifffile import TiffFile,imwrite
from skimage.morphology import label
from skimage.segmentation.morphsnakes import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set,circle_level_set)
from skimage.morphology import (remove_small_objects,
                                reconstruction, skeletonize)
# https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb

from skimage.segmentation import mark_boundaries
from scipy.ndimage.morphology import distance_transform_edt 
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import zoom
from scipy import ndimage

from tifffile import TiffFile,imsave
from skimage import img_as_uint
from sklearn import svm
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns
import time
import logging
import warnings


def _load_image(prm):

    logging.info(f"___> Loading Image...")
    img2D = TiffFile(prm['input_img']).asarray()
    logging.info(f"_______> {prm['input_img']} loaded (shape={img2D.shape})")

    if prm['save_img']:
        imsave(str(prm['log_folder'] / '00_raw.tif'), (img2D))

    return img2D


def _filter_frangi(prm, img2D):
    logging.info(f"___> Applying frangi filter...")
    frangi_img2D = frangi(np.invert(img2D), **prm['filter_arg']['frangi']) 
    if prm['save_img']:
        imsave(str(prm['log_folder'] / "01_frangi.tif"), (frangi_img2D))

    frangi_img2D_embryo = frangi(np.invert(img2D), **prm['filter_arg']['frangi2']) 
    if prm['save_img']:
        imsave(str(prm['log_folder'] / "01_frangi_embryo.tif"), (frangi_img2D_embryo))

    return frangi_img2D, frangi_img2D_embryo 

def _filter_binarize_frangi(prm, frangi_img2D):

    logging.info(f"___> Applying threshold on frangi image")
    logging.info(f"_______>step 1 - locating seeds...")

    prm['filter_arg']['thresholds']['seed'] = threshold_otsu(frangi_img2D) * prm['filter_arg']['thresholds']['otsu_factor']
    seed_img_bin = np.where(frangi_img2D > prm['filter_arg']['thresholds']['seed'] , 1, 0)
    if prm['save_img']:
        imsave(str(prm['log_folder'] / "02_seed_img.tif"), (seed_img_bin.astype(bool)))

    logging.info(f"_______>step 2 - extending seeds...")
    threshold_bottom = prm['filter_arg']['thresholds']['seed'] / prm['filter_arg']['thresholds']['seed_extension_factor']
    seed_img = frangi_img2D.copy()
    seed_img[seed_img < prm['filter_arg']['thresholds']['seed']] = 0
    bin_img = reconstruction(seed=seed_img, mask=frangi_img2D, method='dilation') >= threshold_bottom

    logging.info(f"_______>step 3 - skeletonizing image..")
    bin_img = skeletonize(bin_img)
    if prm['save_img']:
        imsave(str(prm['log_folder'] / '03_bin_img.tif'), (bin_img.astype(bool)))

    return bin_img

def _segment_regions(prm, bin_img):

    logging.info(f"___> Segmenting regions-----------------------------------------------------------")
    logging.info(f"_______>step 1 - calculating distance transform...")
    DT_img = distance_transform_edt(np.invert(bin_img), return_distances=True, return_indices=False) 
    if prm['save_img']:
        imsave(str(prm['log_folder'] / '04_DT_img.tif'), (DT_img))

    logging.info(f"_______>step 2 - locating peaks...")
    DT_peaks_img = np.where(DT_img > prm['filter_arg']['segmentation']['threshold_DT'],1,0)
    if prm['save_img']:
        imsave(str(prm['log_folder'] /  '05_DT_peaks_img.tif'), DT_peaks_img.astype(bool))

    logging.info(f"_______>step 3 - labelling peaks...")
    label_img = ndi.label(DT_peaks_img)[0]
    a_unique_labels = np.unique(label_img)
    logging.info('__________>{0} regions are detected'.format(a_unique_labels.shape[0] - 1 )) 
    if prm['save_img']:
        imsave(str(prm['log_folder'] / '06_label_img.tif'), label_img)

    logging.info(f"_______>step 4 - restore shape via watershed...")
    ws_img = watershed(-DT_img, label_img, mask=DT_img)  
    if prm['save_img']:
        imsave(str(prm['log_folder'] / '07_watershed.tif'), ws_img)

    logging.info(f"_______>step 5 - filter background regions (size filter)...")
    a_count = np.bincount(ws_img.flatten())
    filter_labels_max = np.argwhere(a_count > prm['filter_arg']['segmentation']['size_filter_max'])
    filter_labels_min = np.argwhere(a_count < prm['filter_arg']['segmentation']['size_filter_min'])
    filter_labels = np.append(filter_labels_max.flatten(), filter_labels_min.flatten())

    a_unique_labels = np.unique(label_img)
    logging.info(f"__________>{len(filter_labels)} labels are filtered out using a size filter")
    ws_img_filtered = np.where(np.isin(ws_img, filter_labels), 0, ws_img)
    if prm['save_img']:
        imsave(str(prm['log_folder'] / '08_watershed_filtered.tif'), ws_img)


    return ws_img_filtered


def segment_embryos(prm):
    logging.info(f">Segmenting image into regions...")

    img2D = _load_image(prm)
    frangi_img2D, frangi_img2D_embryo   =_filter_frangi(prm, img2D)
    bin_img = _filter_binarize_frangi(prm, frangi_img2D)
    ws_img_filtered = _segment_regions(prm, bin_img)

    if prm['save_img']:
        img_FN_spotter = np.where(ws_img_filtered > 0, ws_img_filtered, img2D - np.min(img2D))
        imsave(str(prm['log_folder'] / "99_FN_spotter_help.tif"), img_FN_spotter)

        img_FP_spotter = np.where(ws_img_filtered > 0, img2D, 0)
        imsave(str(prm['log_folder'] /"99_FP_spotter_help.tif"), img_FP_spotter)

    d_img = {'img2D': img2D, 'frangi_img2D': frangi_img2D, 'frangi_img2D_embryo': frangi_img2D_embryo, 'img_regions': ws_img_filtered}
    return d_img


def extract_region_properties(prm, l_prop=[['label', 'centroid']], intensity_image_names=None, l_suffix=[''], d_img={}, verbose=False):
    '''https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops'''

    if not verbose:
        prm['logging_root'].setLevel(logging.ERROR)

    logging.info(f"> Extracting properties from regions-----------------------------------------------------------")

    def _load_img_files(prm, d_img):
        logging.info(f"____> Loading images")

        if d_img.get('img_regions') is None:
            d_img['img_regions'] = TiffFile(prm['p_img_regions']).asarray()

        l_intensity_image = None
        if intensity_image_names:
            l_intensity_image = []
            for intensity_image in intensity_image_names:
                if d_img.get(intensity_image) is None:
                    d_img[intensity_image] = TiffFile(prm[f"p_{intensity_image}"]).asarray()
                l_intensity_image.append(d_img[intensity_image])

        return d_img, l_intensity_image

    def get_region_props(img, l_intensity_image=[None], l_prop=None, l_suffix=['']):
        l_df_props= []
        for intensity_image, prop, suffix in zip(l_intensity_image, l_prop, l_suffix):
            if not prop:
                prop = ('label', 'centroid', 'area','mean_intensity')
            df_prop = pd.DataFrame(regionprops_table(img, intensity_image=intensity_image, properties=prop))
            df_prop.columns = [i + suffix for i in df_prop.columns]
            l_df_props.append(df_prop)
        return pd.concat(l_df_props, axis=1)

    d_img, l_intensity_image = _load_img_files(prm, d_img)

    df_regions = get_region_props(
        img=d_img['img_regions'],
        l_intensity_image=l_intensity_image if l_intensity_image else [None],
        l_prop = l_prop,
        l_suffix=l_suffix)

    logging.info(f"____> {len(df_regions.index)} regions segmented in the image")

    prm['logging_root'].setLevel(logging.DEBUG)

    return df_regions


def _save_fig(file_out, verbose=False):
    ax = plt.gca()
    fig = plt.gcf()
    plt.tight_layout()

    plt.savefig(file_out, transparent=False, dpi=350, facecolor='white', edgecolor='none')  # facecolor=fig.get_facecolor()
    if verbose:
        print(file_out, flush=True)
    return


def displot_2D(prm, df_regions, x, y, kind='hist', hue='lbl_pred'):
    d_args = {'x':x, 'y':y, 'kind':kind, 'hue':hue, 'palette': {'P':'g','N':'r','TP':'g','FP':'r','FN':'blue',True:'g',False:'r','RES':'blue', 'VAL':'green'}} # hist or kde
    if d_args['kind'] == 'kde' and not ('y' in d_args):
        d_args['cut'] = 0
    if 'lbl_real' in df_regions:
        d_args['style'] = 'lbl_real'
    if kind == 'dis':
        sns.displot(df_regions, **d_args)
    else:
        del d_args['kind']
        sns.scatterplot(data=df_regions, **d_args, alpha=0.4)
    # logging.info(f"____> count values {[ i for i in df_regions[hue].value_counts().items()]}")

    if prm['save_img']:
        _save_fig(prm['log_folder'] / f"displot_2D({x}_{y}.svg")
    return


def add_PCA_coef(prm, df_regions):
    logging.info(f"> Performing PCA-----------------------------------------------------------")
    pca = PCA(n_components=2)
    X = df_regions[prm['clf']['train_properties']]
    pca.fit(X)
    logging.info(f"____> properties =  {prm['clf']['train_properties']}")
    logging.info(f"____> Explained variance =  {pca.explained_variance_ratio_}")
    logging.info(f"____> PCA singular values =  {pca.singular_values_}")
    a_pca_scores = np.matmul(X, pca.components_.T)
    df_regions['PCA1'] = a_pca_scores[0]
    df_regions['PCA2'] = a_pca_scores[1]

    return


def _check_for_nans():
    for col, ser in df_RES_props.items():
        if sum(ser.isna())>0:
            print(f'{col} contains nans !')
    return

def _train_svm(prm, X, y):
    clf = svm.SVC(kernel=prm['clf']['svm_kernel'], C=1, class_weight=prm['clf']['training_weights'])  # change weights to prefer FP or not
    clf.fit(X,y)
    return clf


def predict_with_svm(prm, clf, df_regions, d_img={}):
    logging.info(f"> Predicting with SVM-----------------------------------------------------------")


    cond = df_regions.type == 'RES'
    X = df_regions[cond][prm['clf']['train_properties']]
    df_regions.loc[cond, 'lbl_pred_svm'] = clf.predict(X)

    return df_regions


def train_svm(prm, df_regions):
    logging.info(f"> Training SVM-----------------------------------------------------------")
    logging.info(f"____> train properties : {prm['clf']['train_properties']}")

    cond = df_regions.type == 'RES'
    X = df_regions[cond][prm['clf']['train_properties']]
    y = df_regions[cond].lbl_real
    clf = _train_svm(prm, X,y)

    return clf



def perform_validation(prm, df_regions=None, img_regions=None):

    logging.info(f"> Performing validation-----------------------------------------------------------")

    def _compose_regions_val():
        img_val = TiffFile(prm['img_val']).asarray()
        img_val_lbl = label(np.all(img_val == prm['val_rgb'], axis=2))
        # prm, l_prop=None, intensity_image_names =None, l_suffix=[''], d_img={}
        df_regions_val = extract_region_properties(prm, d_img={'img_regions' : img_val_lbl})
        df_regions_val[['centroid-0', 'centroid-1']] = df_regions_val[['centroid-0', 'centroid-1']].astype(int)
        df_regions_val['validates_RES_label'] = img_regions[(df_regions_val['centroid-0'], df_regions_val['centroid-1'])]
        df_regions_val['type'] = 'VAL'

        return df_regions_val


    def _validate_regions_val():
        ctr_undersegmentation =0
        ctr_missed_cells =0
        for res_label, df_i in df_regions_val.groupby('validates_RES_label'):
            index_mark=None
            if res_label == 0:
                index_mark = df_i.index
                ctr_missed_cells += len(df_i.index)
            elif len(df_i.index>1):
                index_mark = df_i.index[1:]
                ctr_undersegmentation += len(df_i.index[1:])

            if index_mark is not None:
                df_regions_val.loc[index_mark, 'lbl_real'] = 'P'
                df_regions_val.loc[index_mark, 'lbl_pred'] = 'N'
                if 'lbl_pred_svm' in df_regions:
                    df_regions_val.loc[index_mark, 'lbl_pred_svm'] = 'N'

        return


    def _validate_regions_res():
        key = ['validates_RES_label']
        cols = ['label']
        df_regions_merged  = pd.merge(df_regions, df_regions_val[key + cols].drop_duplicates(subset=key, ignore_index=True, inplace=False), how='left', left_on='label', right_on=key, suffixes=('','_VAL'))
        logging.info(f"____>{len(df_regions.index)} regions were initially selected as P candidates, before filtering by classifier")
        df_regions_merged.rename({'label_VAL':'validated_by_VAL'}, axis=1, inplace=True)
        df_regions_merged.drop(key[0], axis=1, inplace=True)
        cond = df_regions_merged.validated_by_VAL.isna()
        df_regions_merged.loc[cond , 'lbl_real']  = 'N'
        df_regions_merged.loc[~cond , 'lbl_real']  = 'P'

        return df_regions_merged

    df_regions = copy.deepcopy(df_regions[df_regions.type == 'RES'])
    df_regions.drop(['validated_by_VAL', 'validates_RES_label'], axis=1, inplace=True, errors='ignore')

    if img_regions is None:
        img_regions =  TiffFile(prm['p_img_regions']).asarray()
    if df_regions is None:
        df_regions = extract_region_properties(prm, d_img={'img_regions' : img_regions})
        
    df_regions_val = _compose_regions_val()
    _validate_regions_val()
    df_regions = _validate_regions_res()
    
    # return pd.concat([df_regions, df_regions_val], ignore_index=True)
    return pd.concat([df_regions, df_regions_val], ignore_index=True)


def report_count(prm, df_regions, d_img):

    logging.info(f"> Report count-----------------------------------------------------------")

    def compose_img_report():
        from skimage.draw import circle_perimeter
        if d_img.get('img2D') is None:
            d_img['img2D'] = TiffFile(prm['p_img2D']).asarray()
        img2D_vis = d_img['img2D']  - np.min(d_img['img2D'])
        anchor_value = np.max(img2D_vis) 
        cond_RES = df_regions.type == 'RES'

        lbl_pred = 'lbl_pred_svm' if 'lbl_pred_svm' in df_regions else 'lbl_pred'
        cond = df_regions[lbl_pred] == 'P'
        for ix_row, row_i in df_regions[cond_RES & cond].iterrows():
            rr, cc = circle_perimeter(int(row_i['centroid-0']), int(row_i['centroid-1']),radius=5)
            if 'lbl_real' in df_regions:
                marker_value = 0 if row_i['lbl_real']== 'N' else anchor_value
            else:
                marker_value = anchor_value
            img2D_vis [rr, cc] = marker_value


        cond_FN1 = df_regions.type == 'VAL'
        if sum(cond_FN1) > 0:
            cond_FN2 = df_regions.lbl_real == 'P'
            for ix_row, row_i in df_regions[cond_FN1 & cond_FN2].iterrows():
                rr, cc = circle_perimeter(int(row_i['centroid-0']), int(row_i['centroid-1']),radius=10)
                img2D_vis [rr, cc] = anchor_value * 2

        return img2D_vis


    for type_i, df_i in df_regions.groupby('type'):
        if type_i == 'RES':
            metric =  'lbl_pred_svm' if 'lbl_pred_svm' in df_i else 'lbl_pred'
            cond = df_i[metric] == 'P'
            logging.info(f"Embryo count prediction = {sum(cond)} ({metric})")
        elif type_i == 'VAL':
            metric =  'lbl_real'
            cond = df_regions[metric] == 'P'
            logging.info(f"Embryo count (Ground Truth) = {sum(cond)} ({metric})")
            metric =  'lbl_pred_svm' if 'lbl_pred_svm' in df_regions else 'lbl_pred'
            logging.info(pd.crosstab(index=df_regions['lbl_real'], columns=df_regions[metric]))

            cond_FN1 = df_regions.type == 'VAL'
            cond_FN2 = df_regions.lbl_real == 'P'
            cond_missed_cell = df_regions.validates_RES_label == 0
            df_missed = df_regions[cond_FN1 & cond_FN2 & cond_missed_cell]
            df_undersegmentation = df_regions[cond_FN1 & cond_FN2 & ~cond_missed_cell]
            logging.info(f"Nb of missed cells = {len(df_missed.index)}, nb of undersegmented cells = {len(df_undersegmentation.index)}")

    if prm['save_img']:
        imsave(str(prm['output_folder_file'] /"report.tif"), compose_img_report())

    return
        


def filter_regions(prm, df_regions):

    logging.info(f"> Filter regions----------------------------------------------------------")

    cond_RES = df_regions.type=='RES'
    filter1 = df_regions.major_axis_length.between(*prm['filter_region']['major_axis_length_interval'])
    filter2 = df_regions.minor_axis_length.between(*prm['filter_region']['minor_axis_length_interval'])
    filter3 = df_regions.area.between(*prm['filter_region']['area_interval'])
    cond = cond_RES & (filter1 | filter2) & filter3
    logging.info(f"___> {sum(~cond)} regions are filtered based on axis characteristics and area,leaving {sum(cond)} regions")

    df_regions.loc[~cond, 'lbl_pred'] = 'N'
    df_regions.loc[cond, 'lbl_pred'] = 'P'

    return df_regions



