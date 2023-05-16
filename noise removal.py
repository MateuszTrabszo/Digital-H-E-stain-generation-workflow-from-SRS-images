import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import io
import cv2
import tifffile



image = (io.imread("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/HE.tif ")).astype('float64')
#Need to convert to float as we will be doing math on the array

#Non-local means filtering:

sigma_est = np.mean(estimate_sigma(image, multichannel=True))



patch_kw = dict(patch_size=7,      
                patch_distance=6,  
                multichannel=True)

result = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=7, patch_distance=6, multichannel=True)

#convert to 16-bit
result = (result.astype('uint16'))
#saves as tiff
tifffile.imwrite("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/HE_nlm2.tif ", result, photometric="rgb")

from scipy.ndimage import median_filter

image2 = (io.imread("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/HE_nlm2.tif ")).astype('float64')

# Assuming that the 4-channel array is stored in a variable called "image"
# The dimensions of the image can be obtained using the "shape" attribute
image_height, image_width, channels = image2.shape


# Split the 4-channel image into 4 separate channels
channel_0 = image2[:, :, 0]
channel_1 = image2[:, :, 1]
channel_2 = image2[:, :, 2]

# Apply median filter to each channel separately
channel_0_filtered = median_filter(channel_0, size=3)
channel_1_filtered = median_filter(channel_1, size=3)
channel_2_filtered = median_filter(channel_2, size=3)

# Combine the filtered channels back into a 4-channel image
filtered_image = np.zeros((image_height, image_width, channels))
filtered_image[:, :, 0] = channel_0_filtered
filtered_image[:, :, 1] = channel_1_filtered
filtered_image[:, :, 2] = channel_2_filtered

#conberts to 16-bit
filtered_image = filtered_image.astype('uint16')

#saves as a tiff
tifffile.imwrite("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/HE_filtered2.tif", filtered_image, photometric="rgb")
