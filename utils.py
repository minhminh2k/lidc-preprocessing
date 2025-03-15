import argparse
import os
import cv2
import numpy as np

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
    # image = img.copy()
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
    return mask*img

def make_lungmask(img):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
    
    return mask * img # , mask: Lung Image, Mask


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def resize_image(image: np.ndarray, size: int = 256, interpolation=cv2.INTER_CUBIC):
    resized_image = cv2.resize(image, (size, size), interpolation=interpolation)
    return resized_image

def resize_mask(mask: np.ndarray, size: int = 256, interpolation=cv2.INTER_CUBIC):
    resized_mask = cv2.resize(mask.astype(float), (size, size), interpolation=interpolation)
    resized_mask = resized_mask.astype(bool)
    return resized_mask

def ct_normalize(image, slope, intercept):
    image[image==-0] = 0
    # image = image * slope + intercept # Convert to HU 
    # image[image > 400] = 400
    # image[image < -1000] = -1000
    return image

def HU_conversion(scans):
    image = np.stack([s.pixel_array for s in scans]) # axis = 0
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    image[image==-0] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

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
    return t


def resample(image, scan, new_spacing=[1,1,1]) -> np.ndarray:
    # Determine current pixel spacing
    spacing = map(float, ([scan.slice_thickness] + [scan.pixel_spacing] + [scan.pixel_spacing]))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

    
def normalize_clipped(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1.0] = 1.0
    image[image<0.0] = 0.0
    return image


def zero_center(image, PIXEL_MEAN = 0.25): # LUNA16: Mean ALL datasets = 0.25
    image = image - PIXEL_MEAN
    return image


# Lung Segmentation: https://www.kaggle.com/code/akh64bit/full-preprocessing-tutorial/notebook#Normalization
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True): # include structures within the lung
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

# segmented_lungs = segment_lung_mask(pix_resampled, False)
# segmented_lungs_fill = segment_lung_mask(pix_resampled, True) # better

# Get 3D Cube
def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res