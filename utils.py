import argparse
import os
import numpy as np
import cv2
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    image = img.copy()
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask*image

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def resize_image(image: np.ndarray, size: int = 512, interpolation=cv2.INTER_CUBIC):
    resized_image = cv2.resize(image, (size, size), interpolation=interpolation)
    return resized_image

def resize_mask(mask: np.ndarray, size: int = 512, interpolation=cv2.INTER_CUBIC):
    resized_mask = cv2.resize(mask.astype(float), (size, size), interpolation=interpolation)
    resized_mask = resized_mask.astype(bool)
    return resized_mask

def ct_normalize(image, slope, intercept):
    image[image==-0] = 0
    # image = image * slope + intercept # Convert to HU 
    # image[image > 400] = 400
    # image[image < -1000] = -1000
    return image

def HU_conversion(image, slope, intercept):
    pass

def padding_tensor(t):
    padding_needed = 128 - t.shape[0]
    padding_left = padding_needed // 2
    padding_right = padding_needed - padding_left
    
    if padding_left == 0:
        t = np.concatenate((t, padding_right * [t[-1]]), axis=0)
    elif padding_right == 0:
        t = np.concatenate((padding_left * [t[0]], t), axis=0)
    else:
        t = np.concatenate((padding_left * [t[0]], t, padding_right * [t[-1]]), axis=0)
    return t, padding_left, padding_right
