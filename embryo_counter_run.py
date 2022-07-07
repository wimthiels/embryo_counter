"""
Created on 13 June 2022
count C. elegans embryos on a plate
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
pdb.set_trace()
from embryo_counter import *
import logging


def _init_logger(log_out):
    logging.basicConfig(filename=log_out, level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    logging.getLogger('matplotlib.font_manager').disabled = True
    return root # logging


def compose_process_steps(prm):
    if prm['mode'] == 'predict':
        # prm['process_steps'] = ['segment', 'extract_region_properties', 'predict_with_svm', 'report']
        prm['process_steps'] = ['segment', 'extract_region_properties', 'filter_regions', 'relabel_hierarchically', 'report']
        # prm['process_steps'] = ['extract_region_properties', 'filter_regions', 'relabel_hierarchically', 'report']
        # prm['process_steps'] = ['cluster_regions', 'extract_region_properties', 'filter_regions', 'report']
    elif prm['mode'] == 'validate':
        prm['process_steps'] = ['validate', 'report']
    elif prm['mode'] == 'train':
        prm['process_steps'] = ['report', 'train_svm', 'predict_with_svm', 'report']

    if prm['input_type'] == 'segmented_images':
        prm['process_steps'] = [i for i in prm['process_steps'] if i not in ['segment']]
    if prm['input_type'] == 'csv':
        prm['process_steps'] = [i for i in prm['process_steps'] if i not in ['segment', 'extract_region_properties']]

    return 


def _init_parms(arg):

    def derive_parms(verbose=True):
        # the standard allowed gap between cells is heuristically set at 1/2 of minor axis
        prm['filter_arg']['segmentation']['threshold_DT'] = (prm['major_axis_length_standard_micron'] * prm['axis_length_ratio_standard'] * prm['factor_tolerate_gap_between_cells'])/ (3*prm['res_XY'] )
        prm['filter_region'] = {}
        major_axis_length_standard_pixel  = prm['major_axis_length_standard_micron'] / prm['res_XY'] 
        major_axis_variance_pixel = prm['major_axis_variance_micron'] / prm['res_XY']
        minor_axis_length_standard_pixel = major_axis_length_standard_pixel * prm['axis_length_ratio_standard']
        minor_axis_variance_pixel  = major_axis_variance_pixel  * prm['axis_length_ratio_standard']
        prm['filter_region']['major_axis_length_interval'] = [major_axis_length_standard_pixel - major_axis_variance_pixel , major_axis_length_standard_pixel + major_axis_variance_pixel]
        prm['filter_region']['minor_axis_length_interval'] = [minor_axis_length_standard_pixel - minor_axis_variance_pixel , minor_axis_length_standard_pixel + minor_axis_variance_pixel]
        prm['filter_region']['area_interval'] = [ i*j*3.14/4 for i, j in zip(prm['filter_region']['major_axis_length_interval'],prm['filter_region']['minor_axis_length_interval'])]
    
        if verbose:
            logging.info(f">Derived parms")
            logging.info(f">___threshold_DT set to {prm['filter_arg']['segmentation']['threshold_DT']}")
            logging.info(f">___Allowed major_axis_length_range set to {prm['filter_region']['major_axis_length_interval']}")
            logging.info(f">___Allowed minor axis_length_range set to {prm['filter_region']['minor_axis_length_interval']}")
            logging.info(f">___Allowed area range set to {prm['filter_region']['area_interval']}")
        return

    prm={}
    prm['mode'] = arg[1] if len(arg) > 1 else 'predict'
    assert (prm['mode'] in ['predict', 'validate', 'train'])
    prm['input_type'] = 'raw' # 'segmented_images'  # 'raw', 'segmented_images', 'csv'
    compose_process_steps(prm)

    prm['root_folder'] = Path("/home/wth/Downloads/testinfra/ISOLATED/labkit_Roberto")
    prm['l_input_img'] = [i for i in (prm['root_folder'] / "INPUT").rglob('*.tif')]
    # prm['l_input_img'] = [prm['root_folder'] / "INPUT" / "B1_Stitching.czi - B1_Stitching.czi #2.tif"]
    # prm['input_img'] = prm['root_folder'] / "INPUT" / "B2_Stitching.czi - B2_Stitching.czi #2.tif"
    # prm['input_img'] = prm['root_folder'] / "INPUT" / "B3_Stitching.czi - B3_Stitching.czi #2.tif"
    # prm['input_img'] = prm['root_folder'] / "INPUT" / "B4_Stitching.czi - B4_Stitching.czi #2.tif"
    prm['img_val'] = prm['root_folder'] / "VALIDATION" / "B1_Stitching.czi - B1_Stitching.czi #2 (validation).tif"
    prm['output_folder'] = prm['root_folder'] / "OUTPUT"
    prm['log_folder'] = prm['output_folder'] / "LOG"

    prm['res_XY'] = 0.454 # keep correct !
    prm['major_axis_length_standard_micron'] = 22.7 # system parameter 
    prm['axis_length_ratio_standard'] = 0.58 # system parameter :
    prm['factor_tolerate_gap_between_cells'] = 0.8 # > 1 more region fragmentation
    prm['major_axis_variance_micron'] = 5 
    
    #----------------------------------------

    compose_process_steps(prm)
    prm['val_rgb'] = [0,255,0] 
    
    prm['p_img2D'] = prm['log_folder'] / '00_raw.tif'
    prm['p_frangi_img2D'] = prm['log_folder'] / "01_frangi.tif"
    prm['p_frangi_img2D_embryo'] = prm['log_folder'] / "01_frangi_embryo.tif"
    prm['p_img_regions'] = prm['log_folder'] / '08_watershed_filtered.tif'
    
    prm['save_img'] = True


    prm['filter_arg'] = {}
    prm['filter_arg']['frangi'] = {
                        'sigmas': [3],   #boundary 6pixels across
                        'scale_step': 1,
                        'alpha': 0.01,  #NA for 2D images
                        'beta': 9999, 
                        'gamma': 0.05,  #0.005
                        'black_ridges': True,
                        'mode':'reflect'}
    prm['filter_arg']['frangi2'] = copy.deepcopy(prm['filter_arg']['frangi'] )
    prm['filter_arg']['frangi2'].update({'sigmas': [7], 'black_ridges': False, 'beta':1})  # 11
    prm['filter_arg']['thresholds'] = {
                        'otsu_factor' : 0.3,
                        'seed_extension_factor': 8
                        }
    prm['filter_arg']['segmentation'] = {
                        # 'threshold_DT' : 13,
                        'size_filter_max' : 4000,
                        'size_filter_min' : 100
                        }

    prm['extract_properties'] = ['label', 'centroid', 'area', 'minor_axis_length', 'major_axis_length', 'eccentricity', 'min_intensity', 'max_intensity', 'mean_intensity', 'feret_diameter_max','bbox']

    prm['clf']={}
    prm['clf']['p_svm_train'] = prm['log_folder'] / 'svm.pkl'
    prm['clf']['p_svm_predict'] = prm['clf']['p_svm_train']
    prm['clf']['train_properties'] = ['area', 'min_intensity']
    # prm['clf']['train_properties'] = ['area', 'minor_axis_length', 'major_axis_length']
    # prm['clf']['train_properties'] = ['area', 'minor_axis_length', 'major_axis_length','min_intensity','mean_intensity_frangi']
    prm['clf']['training_weights'] = {'P':5, 'N':1}
    prm['clf']['svm_kernel'] = 'rbf'

    prm['log_folder'].mkdir(parents=True,exist_ok=True)
    prm['logging_root'] = _init_logger(prm['log_folder']  / 'embryo_counter.log')
    # prm['logging_root'].setLevel(logging.DEBUG)

    logging.info(f"Start logging")
    logging.info(f"<<<Running in {prm['mode']}-MODE>>>")
    logging.info(f"process_steps -> {prm['process_steps']}")

    derive_parms()

    return prm


def set_file_paths(prm, file_i):
    prm['input_img'] = file_i
    prm['output_folder_file'] = prm['output_folder'] / file_i.stem.replace(" ", "")
    prm['p_df_regions_out'] = prm['output_folder_file'] /'regions.csv'
    prm['p_df_regions_in'] = prm['p_df_regions_out']

    prm['output_folder_file'].mkdir(parents=True,exist_ok=True)

    prm['log_folder'] = prm['output_folder_file']

    prm['p_img2D'] = prm['log_folder'] / '00_raw.tif'
    prm['p_frangi_img2D'] = prm['log_folder'] / "01_frangi.tif"
    prm['p_frangi_img2D_embryo'] = prm['log_folder'] / "01_frangi_embryo.tif"
    prm['p_img_regions'] = prm['log_folder'] / '08_watershed_filtered.tif'

    return


if __name__ == '__main__':
    prm = _init_parms(sys.argv)
    d_img = {}
    df_regions = None
    clf = None

    for ix_file, file_i in enumerate(prm['l_input_img']):
        set_file_paths(prm, file_i)
        logging.info(f"---------------------------------------------------------------------------------------------------------------------------------")
        logging.info(f"PROCESSING file {ix_file + 1} of {len(prm['l_input_img'])} : {prm['input_img']}>>>>>>>>>>>>>>>>>>>>>>>>")
        logging.info(f"---------------------------------------------------------------------------------------------------------------------------------")
        for process_step in prm['process_steps']:
            if process_step == 'segment':
                d_img = segment_embryos(prm)
                continue

            if process_step == 'cluster_regions':
                img_regions = d_img.get('img_regions', TiffFile(prm['p_img_regions']).asarray())
                d_img['img_regions_preCluster'] = img_regions # Backup old region data for reference
                d_img['img_regions'] = cluster_regions(img_regions)
                continue

            if process_step == 'extract_region_properties':
                df_regions = extract_region_properties(
                        prm,
                        intensity_image_names=['img2D', 'frangi_img2D_embryo'],
                        l_prop=[prm['extract_properties'], ['mean_intensity', 'min_intensity', 'max_intensity']],
                        l_suffix=['','_frangi'],
                        d_img=d_img,
                        verbose=True)
                df_regions['type'] = 'RES'
                df_regions['lbl_pred'] = 'P'
                displot_2D(prm, df_regions, x=prm['extract_properties'][3], y=prm['extract_properties'][4], hue='type')

            df_regions = pd.read_csv(prm['p_df_regions_in']) if df_regions is None else df_regions
            if process_step == 'add_PCA_coef':
                add_PCA_coef(prm, df_regions)
                displot_2D(prm, df_regions, x='PCA1', y='PCA2', hue='type')

            elif process_step == 'validate':
                # df_regions = copy.deepcopy(df_regions[df_regions.type == 'RES'])
                df_regions = perform_validation(prm, df_regions, img_regions=d_img.get('img_regions'))
                
            elif process_step ==  'train_svm':
                clf = train_svm(prm, df_regions)
                with open(prm['clf']['p_svm_train'], 'wb') as f:
                    pickle.dump(clf, f)
                    logging.info(f"____> trained svm written to {str(f)}")

            elif process_step == 'predict_with_svm':
                df_regions.lbl_pred_svm = np.NaN
                if clf is None:
                    clf = pickle.load(open( prm['clf']['p_svm_predict'], "rb" ))
                df_regions = predict_with_svm(prm, clf, df_regions, d_img)

            elif process_step == 'filter_regions':
                df_regions = filter_regions(prm, df_regions)

            elif process_step == 'relabel_hierarchically':
                df_regions = hierarchical_labeling(prm, df_regions, d_img)

            elif process_step == 'report':
                report_count(prm, df_regions, d_img)

            if df_regions is not None:
                df_regions.to_csv(prm['p_df_regions_out'], index=False)
                # logging.info(f"> {str(prm['p_df_regions_out'])} written to output")
