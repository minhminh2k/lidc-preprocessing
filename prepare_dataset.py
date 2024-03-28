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
from pydicom.data import get_testdata_file
from utils import is_dir_path,segment_lung, resize_image, resize_mask, ct_normalize, padding_tensor
from pylidc.utils import consensus
from PIL import Image
import cv2

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
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
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        CHECK_LIST = [
            "LIDC-IDRI-0015",
            "LIDC-IDRI-0019",
            "LIDC-IDRI-0025",
            "LIDC-IDRI-0066",
            "LIDC-IDRI-0072",
            "LIDC-IDRI-0094",
            "LIDC-IDRI-0180",
            "LIDC-IDRI-0280",
            "LIDC-IDRI-0299",
            "LIDC-IDRI-0339",
            "LIDC-IDRI-0340",
            "LIDC-IDRI-0352",
            "LIDC-IDRI-0369",
            "LIDC-IDRI-0371",
            "LIDC-IDRI-0376",
            "LIDC-IDRI-0384",
            "LIDC-IDRI-0385",
            "LIDC-IDRI-0411",
            "LIDC-IDRI-0432",
            "LIDC-IDRI-0438",
            "LIDC-IDRI-0443",
            "LIDC-IDRI-0463",
            "LIDC-IDRI-0467",
            "LIDC-IDRI-0468",
            "LIDC-IDRI-0478",
            "LIDC-IDRI-0488",
            "LIDC-IDRI-0498",
            "LIDC-IDRI-0532",
            "LIDC-IDRI-0537",
            "LIDC-IDRI-0538",
            "LIDC-IDRI-0543",
            "LIDC-IDRI-0545",
            "LIDC-IDRI-0551",
            "LIDC-IDRI-0553",
            "LIDC-IDRI-0554",
            "LIDC-IDRI-0562",
            "LIDC-IDRI-0572",
            "LIDC-IDRI-0591",
            "LIDC-IDRI-0606",
            "LIDC-IDRI-0613",
            "LIDC-IDRI-0621",
            "LIDC-IDRI-0633",
            "LIDC-IDRI-0639",
            "LIDC-IDRI-0648",
            "LIDC-IDRI-0650",
            "LIDC-IDRI-0657",
            "LIDC-IDRI-0658",
            "LIDC-IDRI-0664",
            "LIDC-IDRI-0669",
            "LIDC-IDRI-0684",
            "LIDC-IDRI-0696",
            "LIDC-IDRI-0698",
            "LIDC-IDRI-0715",
            "LIDC-IDRI-0725",
            "LIDC-IDRI-0734",
            "LIDC-IDRI-0756",
            "LIDC-IDRI-0758",
            "LIDC-IDRI-0769",
            "LIDC-IDRI-0778",
            "LIDC-IDRI-0781",
            "LIDC-IDRI-0794",
            "LIDC-IDRI-0799",
            "LIDC-IDRI-0802",
            "LIDC-IDRI-0805",
            "LIDC-IDRI-0818",
            "LIDC-IDRI-0819",
            "LIDC-IDRI-0821",
            "LIDC-IDRI-0826",
            "LIDC-IDRI-0831",
            "LIDC-IDRI-0856",
            "LIDC-IDRI-0857",
            "LIDC-IDRI-0895",
            "LIDC-IDRI-0908",
            "LIDC-IDRI-0920",
            "LIDC-IDRI-0921",
            "LIDC-IDRI-0922",
            "LIDC-IDRI-0932",
            "LIDC-IDRI-0943",
            "LIDC-IDRI-0950",
            "LIDC-IDRI-0971",
            "LIDC-IDRI-0981",
            "LIDC-IDRI-0982",
            "LIDC-IDRI-0989",
            "LIDC-IDRI-1006",
        ]

        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            if not pid in CHECK_LIST:
                continue
            # from IPython import embed; embed()
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            dicom_path = scan.get_path_to_dicom_files()
            files = []
            for f in os.listdir(dicom_path):
                if f.endswith('.dcm'):
                    files.append(f)
            # rescale_intercept = scan.image_dicom[0].RescaleIntercept
            # rescale_slope = scan.image_dicom[0].RescaleSlope
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
            # Resample to a common voxel spacing of 1 mm in all directions. 
            # This is to make sure that the voxel size is consistent across all patients
            # The pixel values were converted to Hounsfield units
            # from IPython import embed; embed()
            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                nodule_idxes = []
                mask_list = []
                malignancy_list = []
                cancer_label_list = []
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    # cbbox=(slice(0, 512, None), slice(0, 512, None), slice(67, 71, None))
                    # nodule_idxes = list(range(cbbox[2].start,cbbox[2].stop))
                    slices = list(range(cbbox[2].start,cbbox[2].stop))
                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    for nodule_slice in range(mask.shape[2]):
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        if slices[nodule_slice] not in nodule_idxes:
                            nodule_idxes.append(slices[nodule_slice])                            
                            mask_list.append(mask[:,:,nodule_slice])
                            malignancy_list.append(malignancy)
                            cancer_label_list.append(cancer_label)
                        else:
                            mask_list[nodule_idxes.index(slices[nodule_slice])] = np.logical_or(mask_list[nodule_idxes.index(slices[nodule_slice])],mask[:,:,nodule_slice])
                            cancer_label_list[nodule_idxes.index(slices[nodule_slice])] = cancer_label or cancer_label_list[nodule_idxes.index(slices[nodule_slice])]                       
                
                lung_np_tensor = []
                mask_np_tensor = []
                meta_list = []
                nodule_name = "{}/{}_NI001".format(pid,pid[-4:])
                mask_name = "{}/{}_MA001".format(pid,pid[-4:])
                for slice in range(vol.shape[2]):
                    image_path = os.path.join(dicom_path,files[slice])
                    ds = pydicom.dcmread(image_path)
                    intercept = ds.RescaleIntercept
                    slope = ds.RescaleSlope
                    if slice in nodule_idxes:
                        lung_np_array = vol[:,:,slice]
                        lung_np_array = ct_normalize(lung_np_array, slope, intercept)
                        current_index = nodule_idxes.index(slice)

                        meta_list.append([
                            pid[-4:],
                            slice,
                            prefix[slice],
                            nodule_name,
                            mask_name,
                            malignancy_list[current_index],
                            cancer_label_list[current_index],
                            False
                        ])
                        lung_np_tensor.append(resize_image(lung_np_array))
                        mask_np_tensor.append(resize_mask(mask_list[current_index]))
                    else:
                        lung_np_array = vol[:,:,slice]
                        lung_np_array = ct_normalize(lung_np_array, slope, intercept)
                        # meta_list.append([pid[-4:],slice,prefix[slice],nodule_name,mask_name,0,False,True])
                        lung_np_tensor.append(resize_image(lung_np_array))
                        mask_np_tensor.append(np.zeros_like(resize_image(lung_np_array)))

                lung_np_tensor = np.stack(lung_np_tensor, axis=0)
                mask_np_tensor = np.stack(mask_np_tensor, axis=0)
                length = lung_np_tensor.shape[0]
                if length > 128:
                    mask_indices = []
                    for i, m in enumerate(mask_np_tensor):
                        if m.sum() != 0:
                            mask_indices.append(i)
                    # Example setup (for illustration)
                    # lung_np_tensor = np.random.rand(200, 512, 512)  # Your actual lung tensor
                    # mask_np_tensor = np.random.randint(0, 2, (200, 512, 512))  # Your actual mask tensor
                    # Assume length and mask_indices are defined as in your snippet

                    # Determine the essential range to include
                    min_mask_index = min(mask_indices) if mask_indices else 0
                    max_mask_index = max(mask_indices) if mask_indices else length

                    # Determine the desired crop size
                    desired_crop_size = 128  # Example: 128 slices

                    # Calculate center point of the mask indices to try centering the crop around this region
                    mask_center = (min_mask_index + max_mask_index) // 2

                    # Calculate the start and end points of the crop
                    crop_start = max(0, min(mask_center - desired_crop_size // 2, length - desired_crop_size))
                    crop_end = min(length, max(mask_center + desired_crop_size // 2, desired_crop_size))

                    # Adjust if the essential mask indices span is larger than the crop size
                    if (max_mask_index - min_mask_index) > desired_crop_size:
                        # Here you might need to decide on a policy, for example, expand the crop to include all or just move the window
                        crop_start = min_mask_index
                        crop_end = min(length, max_mask_index)
                        if (crop_end - crop_start) < desired_crop_size:  # If possible, expand to desired crop size
                            expansion = desired_crop_size - (crop_end - crop_start)
                            crop_start = max(0, crop_start - expansion // 2)
                            crop_end = min(length, crop_end + expansion // 2)

                    # Apply the crop
                    lung_np_tensor = lung_np_tensor[crop_start:crop_end, :, :]
                    mask_np_tensor = mask_np_tensor[crop_start:crop_end, :, :]
                    for meta in meta_list:
                        meta[1] = meta[1] - (length // 2 - 64)
                        meta[2] = prefix[meta[1]]
                        self.save_meta(meta)

                elif length < 128:
                    # padding
                    lung_np_tensor, padding_left, padding_right = padding_tensor(lung_np_tensor)
                    mask_np_tensor, padding_left, padding_right = padding_tensor(mask_np_tensor)
                    for meta in meta_list:
                        meta[1] = meta[1] + padding_left
                        meta[2] = prefix[meta[1]]
                        self.save_meta(meta)

                np.save(IMAGE_DIR / nodule_name, lung_np_tensor)
                np.save(MASK_DIR / mask_name, mask_np_tensor) 
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                lung_np_tensor = []
                nodule_name = "{}/{}_CN001".format(pid,pid[-4:])
                mask_name = "{}/{}_CM001".format(pid,pid[-4:])
                for slice in range(vol.shape[2]):
                    image_path = os.path.join(dicom_path,files[slice])
                    ds = pydicom.dcmread(image_path)
                    intercept = ds.RescaleIntercept
                    slope = ds.RescaleSlope

                    lung_segmented_np_array = vol[:,:,slice]
                    lung_segmented_np_array = ct_normalize(lung_segmented_np_array, slope, intercept)
                    lung_np_tensor.append(resize_image(lung_segmented_np_array))

                    #CN= CleanNodule, CM = CleanMask
                    # meta_list = [pid[-4:],slice,prefix[slice],nodule_name,mask_name,0,False,True]

                lung_np_tensor = np.stack(lung_np_tensor, axis=0)
                length = lung_np_tensor.shape[0]
                if length > 128:
                    # center crop
                    lung_np_tensor = lung_np_tensor[length // 2 - 64:length // 2 + 64,:,:]
                elif length < 128:
                    # padding
                    lung_np_tensor, _, _ = padding_tensor(lung_np_tensor)
                np.save(CLEAN_DIR_IMAGE / nodule_name, lung_np_tensor)
                np.save(CLEAN_DIR_MASK / mask_name, np.zeros_like(lung_np_tensor))


        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()


    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()
