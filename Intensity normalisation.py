import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import tifffile

#Loading image
im = Image.open('C:/Users/mt598/OneDrive - University of Exeter/Mphys Project/10-10-22 (1)/5liver_CH2 (1).tif')
#converts image to an array
img = np.array(im)

#Prints original image
plt.figure(figsize = (20,12))
plt.imshow(img, cmap = 'gray')
plt.axis('off')
#plt.savefig('original.jpg', bbox_inches ='tight') # uncomment to save figure
plt.show()
plt.close()

#Converts image to a float32 datatype. 
IMG = np.float32(img).copy()

#Defines the number of x and y
x, y = IMG.shape


#Set the number of tiles
x_Tiles = 13
y_Tiles = 14

# Calculates the total number of tiles
Tiles = y_Tiles * x_Tiles

#Calculate the length of each tile
Y_Tile = int(y / y_Tiles)
X_Tile =  int(x / x_Tiles)




#Creates a minimum threshold intensity for defining foreground objects
threshold = 75

#Constructs a foreground image, taking out intensities below a threshold, defined above
IMG_foreground = IMG.copy()
IMG_foreground[IMG_foreground < threshold] = 0
        
#Prints foreground objects image
plt.figure(figsize = (20,12))
plt.imshow(IMG_foreground, cmap = 'gray')
plt.axis('off')
#plt.savefig('original.jpg', bbox_inches ='tight')
plt.show()
plt.close()

#Make a matrix m(x,y) to be the average tile, H_xy to be an average H tile that can divide by
m_xy = np.zeros((X_Tile, Y_Tile), int)
H_xy = np.zeros((X_Tile, Y_Tile), int)

#Define a matrix for H function
H = np.zeros((x,y), int)

#Define the H function for foreground objects
H[IMG_foreground > 0] = 1

#Calculates m(x,y)
for i in range(X_Tile):
   for j in range(Y_Tile):
       for k in range(x_Tiles):
           for l in range(y_Tiles):
               m_xy[i,j] = m_xy[i,j] + IMG_foreground[i + (k*(Y_Tile)), j + (l*(X_Tile))]
               H_xy[i,j] = H_xy[i,j] + H[i + (k*(Y_Tile)), j + (l*(X_Tile))]
m_xy = m_xy/H_xy



#Seperating m(x,y) int seperate vectors
m_x = (m_xy.sum(axis = 1))/ X_Tile
m_y = (m_xy.sum(axis = 0))/ Y_Tile



#Finds m_avg
m_avg = 0.5* (((m_x.sum() / X_Tile)) + (m_y.sum() / Y_Tile))


#Smoothinf of vectors using a gaussian filter
s_x = m_x.copy()
w_x = gaussian_filter1d(m_x, 6, truncate = 2 , output = s_x)

s_y = m_y.copy()
w_y = gaussian_filter1d(m_y, 6, truncate = 2 , output = s_y)


#Can set the edge of the new vectors to be the same value as the original vector
edge = 3
w_x_size = (len(w_x) -1)
w_y_size = (len(w_y) -1)

for i in range(edge):
    w_x[i] = m_x[i]
    w_x[w_x_size - i] = m_x[w_x_size - i]
    w_y[i] = m_y[i]
    w_y[w_y_size - i] = m_y[w_y_size - i] 
    

#Display the original and smoothed vectors on a graph

plt.plot(m_x, color ='r', label='original data')
plt.plot(w_x, 'w', color = 'g' ,label='filtered data')
plt.title("A comparison of the original data, and Gaussian filtered data for the pixel intensity entries of the m_x array")
plt.ylabel('Pixel intensity')
plt.xlabel('m_x array entry')
plt.legend()
plt.grid()
plt.savefig('C:/Users/mt598/OneDrive - University of Exeter/Mphys Project/10-10-22 (1)/5liver_CH2 (1) m_x graph.jpg')
plt.show()

plt.plot(m_y, color ='r', label='original data')
plt.plot(w_y, 'w', color = 'g' ,label='filtered data')
plt.title("A comparison of the original data, and Gaussian filtered data for the pixel intensity entries of the m_y array")
plt.ylabel('Pixel intensity')
plt.xlabel('m_y array entry')
plt.legend()
plt.grid()
plt.savefig('C:/Users/mt598/OneDrive - University of Exeter/Mphys Project/10-10-22 (1)/5liver_CH2 (1) m_y graph.jpg')
plt.show()

#divide vectors by average of m
s_xx = s_x / m_avg
s_yy = s_y / m_avg

#performs outer product of vectors
s_xy = np.outer(s_xx,s_yy)

#generates the correction map for each tile
c_xy = 1/ s_xy

C = img.copy()

#Calculates C(x,y), the corrected image
for i in range(X_Tile):
   for j in range(Y_Tile):
       for k in range(x_Tiles):
          for l in range(y_Tiles):
               C[(i+(k*Y_Tile)), (j +(l*X_Tile) )] = ( C[ (i +(k*Y_Tile)) , ( j +(l*X_Tile) )] *  c_xy[i,j] )

#saves the corrected image as a tiff
tifffile.imwrite('C:/Users/mt598/OneDrive - University of Exeter/Mphys Project/10-10-22 (1)/5liver_CH2 (1)normalised.tif', C)


#Prints the corrected image
plt.figure(figsize = (20,12))
plt.imshow(C, cmap = 'gray')
plt.axis('off')
plt.show()
plt.close()

#Prints the correction intensity map
plt.figure(figsize = (20,12))
plt.imshow(c_xy, cmap = 'gray')
plt.savefig('C:/Users/mt598/OneDrive - University of Exeter/Mphys Project/10-10-22 (1)/5liver_CH2 (1) C_xy.jpg')
plt.axis('off')
plt.show()
plt.close()


