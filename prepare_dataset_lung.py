import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
import pydicom
from utils import is_dir_path,segment_lung, resize_image, resize_mask, padding_tensor
from utils import segment_lung_mask
from utils import HU_conversion, resample
from utils import make_lungmask_v2
from pylidc.utils import consensus

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
LUNG_SEGMENTATION_DIR_V2 = is_dir_path(parser.get('prepare_dataset','LUNG_SEGMENTATION_PATH_V2'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, LUNG_SEGMENTATION_DIR_V2, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.lung_segmentation_dir_v2 = LUNG_SEGMENTATION_DIR_V2
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])


    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        # self.meta = self.meta.append(tmp,ignore_index=True)
        self.meta = pd.concat([self.meta, pd.DataFrame([tmp])], ignore_index=True)
    
    def prepare_dataset(self):
        # Make directory
        if not os.path.exists(self.lung_segmentation_dir_v2):
            os.makedirs(self.lung_segmentation_dir_v2)

        LUNG_SEGMENTATION_DIR_V2 = Path(self.lung_segmentation_dir_v2)

        for patient in tqdm(self.IDRI_list):
            pid = patient # LIDC-IDRI-0001~

            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first() # If needed: pl.Scan.slice_thickness <= 1
            nodules_annotation = scan.cluster_annotations()
            
            print("Patient ID: {} Number of Annotated Nodules: {}".format(pid,len(nodules_annotation)))
            list_scans = scan.load_all_dicom_images() # Instance Number: max_length -> 1 -> max_length.com -> 1.dcom # Sort by ImagePositionPatient z coordinates
            print("Slice Thickness, Scan Slice Thickness: ", list_scans[0].SliceThickness, scan.slice_thickness)
            print("Pixel spacing, Slice Spacing: ", scan.pixel_spacing, scan.slice_spacing)            
            
            # Standardize slice thickness
            for l in list_scans:
                l.SliceThickness = scan.slice_thickness
                        
            original_vol = HU_conversion(list_scans)
            original_vol_shape = original_vol.shape
            print("Original shape: ", original_vol.shape)
            vol, new_spacing = resample(original_vol, scan, [1,1,1])
            print("Resample shape, New spacing:", vol.shape, new_spacing)         
                        
            # Resample to a common voxel spacing of 1 mm in all directions. 
            # This is to make sure that the voxel size is consistent across all patients
            # The pixel values were converted to Hounsfield units
            
            patient_segmentation_dir = LUNG_SEGMENTATION_DIR_V2 / pid
            Path(patient_segmentation_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                nodule_idxes = []
                mask_list = []
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    slices = list(range(cbbox[2].start,cbbox[2].stop))
                    
                    for nodule_slice in range(mask.shape[2]):
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        if slices[nodule_slice] not in nodule_idxes:
                            nodule_idxes.append(slices[nodule_slice])                            
                            mask_list.append(mask[:,:,nodule_slice])
                        else:
                            mask_list[nodule_idxes.index(slices[nodule_slice])] = np.logical_or(mask_list[nodule_idxes.index(slices[nodule_slice])],mask[:,:,nodule_slice])
                
                segmentation_np_tensor = []
                mask_np_tensor = []
                # meta_list = []
                nodule_name = "{}/{}_NI001".format(pid,pid[-4:])
                mask_name = "{}/{}_MA001".format(pid,pid[-4:])
                print("Nodule idxes", nodule_idxes)
                
                # Resampling Mask
                for mask_slice in range(original_vol_shape[0]):
                    if mask_slice in nodule_idxes:
                        current_index = nodule_idxes.index(mask_slice)
                        mask_np_tensor.append(mask_list[current_index])
                    else:
                        mask_np_tensor.append(np.zeros_like(original_vol[mask_slice,:,:]))
                
                # Mask Volume
                mask_volume = np.stack([m for m in mask_np_tensor])
                mask_volume, new_spacing = resample(mask_volume, scan, [1, 1, 1])
                mask_np_tensor = []
                
                # Volume Depth
                length = vol.shape[0]
                
                lung_segmented_original_volume = segment_lung_mask(image=vol, fill_lung_structures=True)
                lung_segmented_processed_volume = make_lungmask_v2(images=vol, thresh_volume=lung_segmented_original_volume)
                
                if length > 128:
                    mask_indices = []
                    for i, m in enumerate(mask_volume):
                        if m.sum() != 0:
                            mask_indices.append(i)
                    print("Mask indices after resample", mask_indices)

                    # Determine the essential range to include
                    min_mask_index = min(mask_indices) if mask_indices else 0
                    max_mask_index = max(mask_indices) if mask_indices else (length - 1)
                    
                    # Determine the desired crop size
                    desired_crop_size = 128  # Example: 128 slices
                    margin = 30
                    
                    if max_mask_index - min_mask_index + 1 >= desired_crop_size:
                        crop_start = max(0, min_mask_index - margin)
                        crop_end = min(max_mask_index + margin, length)
                    else:
                        # Calculate center point of the mask indices to try centering the crop around this region
                        mask_center = (min_mask_index + max_mask_index) // 2

                        # Calculate the start and end points of the crop
                        crop_start = max(0, min(mask_center - desired_crop_size // 2, length - desired_crop_size))
                        crop_end = min(length, max(mask_center + desired_crop_size // 2, desired_crop_size))
                        
                        crop_start = max(0, crop_start - margin)
                        crop_end = min(length, crop_end + margin)
                    
                    for slice in range(crop_start, crop_end):
                        print(slice)
                        segmentation_np_tensor.append(resize_image(lung_segmented_processed_volume[slice, :, :]))
                        
                    segmentation_np_tensor = np.stack(segmentation_np_tensor, axis=0)    
                    print(f"Cropped from slice {crop_start} to {crop_end}.")                
                    print("Cropped Lung Image", segmentation_np_tensor.shape)

                elif length < 128:
                    for slice in range(vol.shape[0]):
                        segmentation_np_tensor.append(resize_image(lung_segmented_processed_volume[slice, :, :]))
                    segmentation_np_tensor = np.stack(segmentation_np_tensor, axis=0)
                    
                    # padding
                    segmentation_np_tensor = padding_tensor(segmentation_np_tensor)
                    
                np.save(LUNG_SEGMENTATION_DIR_V2 / nodule_name, segmentation_np_tensor)
                
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_segmentation = LUNG_SEGMENTATION_DIR_V2 / pid
                Path(patient_clean_dir_segmentation).mkdir(parents=True, exist_ok=True)
                                
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                segmentation_np_tensor = []
                nodule_name = "{}/{}_CN001".format(pid,pid[-4:])
                mask_name = "{}/{}_CM001".format(pid,pid[-4:])
                
                lung_segmented_original_volume = segment_lung_mask(vol, fill_lung_structures=True)
                lung_segmented_processed_volume = make_lungmask_v2(images=vol, thresh_volume=lung_segmented_original_volume)
                
                for slice in range(lung_segmented_processed_volume.shape[0]): # [depth, x, y]             
                    segmentation_np_tensor.append(resize_image(lung_segmented_processed_volume[slice,:,:]))
                    
                segmentation_np_tensor = np.stack(segmentation_np_tensor, axis=0)
                
                length = segmentation_np_tensor.shape[0]
                if length < 128:
                    # padding
                    segmentation_np_tensor = padding_tensor(segmentation_np_tensor)
                
                np.save(LUNG_SEGMENTATION_DIR_V2 / nodule_name, segmentation_np_tensor)

if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    # Lung Segmentation Dir V2
    test= MakeDataSet(LIDC_IDRI_list,LUNG_SEGMENTATION_DIR_V2,mask_threshold,padding,confidence_level)
    test.prepare_dataset()
