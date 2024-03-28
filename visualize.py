import numpy as np

def visualize(pid: str = 'LIDC-IDRI-0001', slice: int = 86):
    # image_path = "data/Image/{}/{}_NI001_slice{}.npy".format(pid,pid[-4:], slice)
    # mask_e_path = "data/Mask/{}/{}_MA001_slice{}_e.npy".format(pid,pid[-4:], slice)
    # mask_0_path = "data/Mask/{}/{}_MA001_slice{}_0.npy".format(pid,pid[-4:], slice)
    # mask_1_path = "data/Mask/{}/{}_MA001_slice{}_1.npy".format(pid,pid[-4:], slice)
    # mask_2_path = "data/Mask/{}/{}_MA001_slice{}_2.npy".format(pid,pid[-4:], slice)
    # mask_3_path = "data/Mask/{}/{}_MA001_slice{}_3.npy".format(pid,pid[-4:], slice)
    
    image_path_pgl = f"data/Image/{pid}/{pid[-4:]}_NI001.npy"
    mask_path_pgl = f"data/Mask/{pid}/{pid[-4:]}_MA001.npy"
    
    # image_path_pgl = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/data/Image/LIDC-IDRI-0001/0001_NI001.npy"
    # mask_path_pgl = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/data/Mask/LIDC-IDRI-0001/0001_MA001.npy"
    
    image_pgl = np.load(image_path_pgl)
    mask_pgl = np.load(mask_path_pgl)
    
    # print(image_pgl.shape, mask_pgl.shape)
    # print(image_pgl[slice].max(), image_pgl[slice].min())
    
    # image = np.load(image_path)
    # print(image.max(), image.min())
    # image = (image - image.min()) / (image.max() - image.min())
    # print(image.max(), image.min())
    
    # mask_e = np.load(mask_e_path)
    # mask_0 = np.load(mask_0_path)
    # mask_1 = np.load(mask_1_path)
    # mask_2 = np.load(mask_2_path)
    # mask_3 = np.load(mask_3_path)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 12))

    row = 3
    col = 3
    
    # plt.subplot(row, col, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Image')

    # plt.subplot(row, col, 2)
    # plt.imshow(mask_e, cmap='gray')
    # plt.title('Mask-ensemble')

    # plt.subplot(row, col, 3)
    # plt.imshow(mask_0, cmap='gray')
    # plt.title('Mask-0')

    # plt.subplot(row, col, 4)
    # plt.imshow(mask_1, cmap='gray')
    # plt.title('Mask-1')

    # plt.subplot(row, col, 5)
    # plt.imshow(mask_2, cmap='gray')
    # plt.title('Mask-2')

    # plt.subplot(row, col, 6)
    # plt.imshow(mask_3, cmap='gray')
    # plt.title('Mask-3')
    
    plt.subplot(row, col, 7)
    plt.imshow(image_pgl[slice], cmap='gray')
    plt.title('Image-PGL')
    
    plt.subplot(row, col, 8)
    plt.imshow(mask_pgl[slice], cmap='gray')
    plt.title('Mask-PGL')

    plt.savefig("image.png")

if __name__ == "__main__":
    visualize("LIDC-IDRI-0010", slice=19)