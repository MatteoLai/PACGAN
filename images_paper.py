import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

root = 'data/ADNI/ADNI_test/Real256x256.nii.gz'
# Real idxs: CN: 17, 51, 121, 127, 393, 407, 416, 436       | AD: 32, 112, 154, 169, 185, 200, 455

# root = "C:/Users/User/Desktop/OneDrive - Alma Mater Studiorum Università di Bologna/GANs for NeuroImaging/AAA_PaperGANs/Results/Bernadette_clean_data/5_100_3_[1, 1, 1, 1, 1, 2, 2]/saved_images/PACGAN_256x256/Generated256x256.nii.gz"
# Fake idxs: CN: 10, 27, 30, 44, 51, 143 | AD: 271, 274, 290, 299, 301, 331, 417, 478, 482,

type = 'Real'
group = 'AD'
sel_idx = [0]

save_root = './RealBrainMRI.nii.gz'#'C:/Users/User/Desktop/OneDrive - Alma Mater Studiorum Università di Bologna/GANs for NeuroImaging/AAA_PaperGANs/Immagini paper'
save_path = save_root + '/' + type + group + str(sel_idx) + '.jpg'

img = nib.load(root)
img = np.asanyarray(img.dataobj) # numpy array of shape (W,H,C,N)
img = np.float32(img)
img = img[:,:,0,sel_idx]


img=np.swapaxes(img, 0,1)

plt.imshow(img, origin='lower', cmap='gray', vmin = -1, vmax = 1)
# plt.savefig(save_path)
mpimg.imsave(save_path, img, origin='lower', cmap='gray', vmin = -1, vmax = 1)
