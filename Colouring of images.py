import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import median_filter
import tifffile

# Load CSV filevfor E data
E_data = pd.read_csv('E_data.csv')

# Convert RGB values to floats in range [0, 1]
E_rgb = E_data[['R', 'G', 'B']] / 255.0

# Create a ListedColormap object from RGB values
Ecmap = ListedColormap(E_rgb.values)

# Display color map
fig, ax = plt.subplots(figsize=(10, 1))
ax.imshow([range(len(E_rgb))], cmap=Ecmap, aspect='auto')
ax.set_xticks(range(len(E_rgb)))
ax.set_yticks([])
#plt.savefig('Ecmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Load CSV file for H data
H_data = pd.read_csv('H_data.csv')

# Convert RGB values to floats in range [0, 1]
H_rgb = H_data[['R', 'G', 'B']] / 255.0

# Create a ListedColormap object from RGB values
Hcmap = ListedColormap(H_rgb.values)

# Display color map
fig, ax = plt.subplots(figsize=(10, 1))
ax.imshow([range(len(H_rgb))], cmap=Hcmap, aspect='auto')
ax.set_xticks(range(len(H_rgb)))
ax.set_yticks([])
#plt.savefig('Hcmap.png', dpi=300, bbox_inches='tight')
plt.show()


#reading images
CH2 = Image.open('C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/CH2_norm.tif')
#reading images
CH3 = Image.open('C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/CH3_norm.tif')

#Converts images to array
CH2_arr = np.array(CH2)

# Compute the histogram
hist_CH2, bins_CH2 = np.histogram(CH2_arr.flatten(), 65536, [0, 65535])

# Compute the cumulative distribution function
cdf_CH2 = hist_CH2.cumsum()

# Normalize the cdf
cdf_norm_CH2 = (cdf_CH2 - cdf_CH2.min()) * 65535 / (cdf_CH2.max() - cdf_CH2.min())

# Compute the equalized image
CH2_eq = np.interp(CH2_arr.flatten(), bins_CH2[:-1], cdf_norm_CH2).reshape(CH2_arr.shape)

# Convert the image to 16-bit unsigned integer
CH2_eq = np.uint16(CH2_eq)

#inverse of the CH2 histogram equalized image
CH2_eqhist_inverse = cv2.bitwise_not(CH2_eq)

#Make a non-equalised array for CH2, and inverse
CH2_not_eqhist = (CH2_arr)
CH2_inverse_not_eqhist = cv2.bitwise_not(CH2_arr)

#Define level of blending between equalised and non-equalised images, higher = more of equalised image
alpha_CH2 = 0.5

#Blend the images using alpha blend factor to create signle CH2 coloured image
CH2_inverse = ((alpha_CH2 * CH2_eqhist_inverse) + ((1 - alpha_CH2) * CH2_inverse_not_eqhist))#.astype('uint16')


#coloured lipid (CH2) signal with Eosin colourmap
CH2_coloured = (Ecmap(CH2_inverse / 65535.0))

#normalize values for 16-bit image
CH2_coloured = ((CH2_coloured *65535)).astype(np.uint16)

#remove the 4th channel (alpha) 
CH2_coloured = CH2_coloured[:,:,:3]

#saves Eosin stained image
tifffile.imwrite("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/E.tif", CH2_coloured, photometric="rgb")

#generate CH3 array 
CH3_arr = np.array(CH3)

# Compute the histogram
hist_CH3, bins_CH3 = np.histogram(CH3_arr.flatten(), 65536, [0, 65535])

# Compute the cumulative distribution function
cdf_CH3 = hist_CH3.cumsum()

# Normalize the cdf
cdf_norm_CH3 = (cdf_CH3 - cdf_CH3.min()) * 65535 / (cdf_CH3.max() - cdf_CH3.min())

# Compute the equalized image
CH3_eq = np.interp(CH3_arr.flatten(), bins_CH3[:-1], cdf_norm_CH3).reshape(CH3_arr.shape)

# Convert the image to 16-bit unsigned integer
CH3_eq = np.uint16(CH3_eq)

#form a difference image (in int32 form) from histogram equalised images
diff =  ((CH3_eq.astype('int64')) - (CH2_eq.astype('int64')) )

#fix difference image data intensity values to be 0 or above
diff[diff<0] = 0

#cobert to 16-bit
diff = diff.astype(np.uint16)

# Compute the histogram
hist_diff, bins_diff= np.histogram(diff.flatten(), 65536, [0, 65535])

# Compute the cumulative distribution function
cdf_diff = hist_diff.cumsum()

# Normalize the cdf
cdf_norm_diff = (cdf_diff - cdf_diff.min()) * 65535 / (cdf_diff.max() - cdf_diff.min())

# Compute the equalized image
diff_eq = np.interp(diff.flatten(), bins_diff[:-1], cdf_norm_diff).reshape(diff.shape)

# Convert the image to 16-bit unsigned integer
diff_eq = np.uint16(diff_eq)


diff_eqhist_inverse = cv2.bitwise_not(diff_eq)

#Make a non-equalised array for CH2, and inverse
diff_not_eqhist = (diff)
diff_inverse_not_eqhist = cv2.bitwise_not(diff_not_eqhist)

#Define level of blending between equalised and non-equalised images, higher = more of equalised image
alpha_diff = 0.6

#Blend the images using alpha blend factor to create signle CH2 coloured image
diff_inverse = ((alpha_diff * diff_eqhist_inverse) + ((1 - alpha_diff) * diff_inverse_not_eqhist))#.astype('uint16')


#coloured lipid (CH2) signal with Eosin colourmap
diff_coloured = (Hcmap(diff_inverse / 65535.0))

#converts to 16-bit
diff_coloured = ((diff_coloured *65535)).astype(np.uint16)

#removes 4th channel
diff_coloured = diff_coloured[:,:,:3]


result = diff_coloured


#Median filter for protein image before the images are joined together


# The dimensions of the image can be obtained using the "shape" attribute
image_height, image_width, channels = result.shape

# Split the 4-channel image into 4 separate channels
channel_0 = result[:, :, 0]
channel_1 = result[:, :, 1]
channel_2 = result[:, :, 2]

# Apply median filter to each channel separately
channel_0_filtered = median_filter(channel_0, size=4)
channel_1_filtered = median_filter(channel_1, size=4)
channel_2_filtered = median_filter(channel_2, size=4)

# Combine the filtered channels back into a 4-channel image
filtered_image = np.zeros((image_height, image_width, channels))
filtered_image[:, :, 0] = channel_0_filtered
filtered_image[:, :, 1] = channel_1_filtered
filtered_image[:, :, 2] = channel_2_filtered


#Diff colored image is filtered image
diff_coloured = filtered_image


#Create a Combined image which is copy of CH2 coloured image as canvas. Take diff image sections to cover it
Combined = CH2_coloured.copy()

#loops for combining images

for i in range(CH2_inverse.shape[0]):
    for j in range(CH2_inverse.shape[1]):
        if (diff_inverse[i,j] < 65535 and CH2_inverse[i,j] > 65534):
            Combined[i,j,:] = diff_coloured[i,j,:]
   
for i in range(CH2_inverse.shape[0]):#w_CH2_a_diff 
    for j in range(CH2_inverse.shape[1]):
        if CH2_inverse[i,j] > 43690 and diff_inverse[i,j] > 16962:
            Combined[i,j,:] = diff_coloured[i,j,:]
            
#tifffile.imwrite("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/T19-68376 I1/HE3.tif ", Combined, photometric="rgb")    
   
for i in range(CH2_inverse.shape[0]):#a_CH2_s_diff 
    for j in range(CH2_inverse.shape[1]):
        if CH2_inverse[i,j] > 21845 and diff_inverse[i,j] < 16962:
            Combined[i,j,:] = diff_coloured[i,j,:]
            
#tifffile.imwrite("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/T19-68376 I1/HE4.tif ", Combined, photometric="rgb")    
 
   
for i in range(CH2_inverse.shape[0]):#s_CH2_w_diff 
    for j in range(CH2_inverse.shape[1]):
        if CH2_inverse[i,j] < 16962 and diff_inverse[i,j] > 43690:
            Combined[i,j,:] = CH2_coloured[i,j,:]

#saves final image as a tiff            
tifffile.imwrite("C:/Users/mt598/OneDrive - University of Exeter/Images for final report/259/HE.tif", Combined, photometric="rgb")    
 