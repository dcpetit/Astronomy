"""
Functions for post-processing.
"""

from typing import Union, Optional, Tuple

import numpy as np

from typeguard import typechecked
from sklearn.decomposition import PCA

from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.sdi import sdi_scaling

#################################################
import cv2
#import cv2
import matplotlib.pyplot as plt

def make_oscillating_checkerboard_noGap(diameter, x, y, center_square, amplitude, amp_nonRad_less, location, delay_x, delay_y, decay_factor): # amp_radial_boost,
    checkerboard4D = np.zeros([x, y, diameter, diameter])
    checkerboard = np.zeros([200, 200])
    for i in range(x):                          #each row of squares
        for j in range(y):                      #each column of squares
            displacement_square = [ i - (center_square[1]-1) , j - (center_square[0]-1) ] # center_square[index] is backward for IMSHOW
            decay_exponent = (np.sqrt(displacement_square[0]**2 + displacement_square[1]**2))
            if displacement_square[1] == 0:
                adjustment = 1 #amp_radial_boost
            else:
                adjustment = amp_nonRad_less**(-1)
            if np.sum(displacement_square) % 2 != 0:
                N = 1 * adjustment * decay_factor**decay_exponent  #if displacement_sqr[0] == 0 and displacement_sqr[1] == 0: N = 0  el
            elif np.sum(displacement_square) % 2 == 0:
                N = -1 * adjustment * decay_factor**decay_exponent
            else:
                print('problem in checkerboard creation')
            for k in range(diameter):                  #each x pixel
                for l in range(diameter):              #each y pixel
                    checkerboard4D[i, j, k, l] = N * amplitude * np.cos(np.pi * k / diameter - delay_y) * np.cos(np.pi * l / diameter - delay_x) # it's an l, not a 1 ... #this one is a sine function, could test it out with a gaussian function
            L0, L1 = location[0], location[1]
            checkerboard[i*diameter+L0:(i+1)*diameter+L0, j*diameter+L1:(j+1)*diameter+L1] = checkerboard4D[i,j,:,:]
    return checkerboard, checkerboard4D
    
def add_wavy_checkerboard(observation, checkerboard):
    merged_observation_WavyCheckerboard = np.add(observation, checkerboard)
    return merged_observation_WavyCheckerboard

def align_CB_to_angle(checkerboard, angle): # np.shape should be [200,200] and Scalar
    import cv2
    # imshow a before checkerboard
    center = tuple(np.array(checkerboard.shape[1::-1]) / 2)
    #print('center should be (100, 100) and is:', center)       # it is, this works
    #print('angle should be +60 and then -60, and is:', angle)  # it is, this works
    if angle < 0: # Or turn into a while loop that adds 360 until it's positive and breaks out of the loop
        #print('the input angle is less than zero... it was,' angle)
        angle = angle + 360
        #print('the angle that will be used for this alignment is now:', angle)
    #print('The angle to make the rotation matrix is:', angle) 
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat_neg = cv2.getRotationMatrix2D(center, np.absolute(angle-360), 1.0)
    #print('\n\nnp.shape(rot_mat), rot_mat:\n', np.shape(rot_mat), rot_mat)  # They are different!
    #print('\nnp.shape(rot_mat_neg), rot_mat_neg:\n', np.shape(rot_mat_neg), rot_mat_neg) #if rot_mat.all() == rot_mat_neg.all(): #True
        #print('Error: The negative angle changes nothing... rot_mat.all() == rot_mat_neg.all()') 
    aligned_checkerboard = cv2.warpAffine(checkerboard, rot_mat, checkerboard.shape[1::-1], flags=cv2.INTER_LINEAR)
    # imshow an after checkerboard
    return aligned_checkerboard, rot_mat

def align_CB_to_angle_inputCenter(checkerboard, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_checkerboard = cv2.warpAffine(checkerboard, rot_mat, checkerboard.shape[1::-1], flags=cv2.INTER_LINEAR)
    return aligned_checkerboard
    
def equal_matrix_test(var1, var2):
    equal_is_zero = 0
    #for i in range(len(np.shape(var1))):
        #for j in range(len(var1[i])): ??? 
    for i in range(len(var1[:,0,0,0])):
        for j in range(len(var1[0,:,0,0])):
            for k in range(len(var1[0,0,:,0])):
                for l in range(len(var1[0,0,0,:])):
                    if var1[i,j,k,l] != var2[i,j,k,l]:
                        print('at indices ('+str(i)+', '+str(j)+', '+str(k)+', '+str(l)+'), the two matrices are not equal')
                        equal_is_zero = 1
    if equal_is_zero == 0:
        print('The two 2-D matrices are EQUAL')
    else:
        print('The two 2-D matrices are UNequal')
    return

def maxF(matrix, WLC, exposure, rounding):
    maximum_in_matrix = str(np.round(np.max(matrix[WLC, exposure,:,:]), rounding))
    return maximum_in_matrix

def minF(matrix, WLC, exposure, rounding):
    minimum_in_matrix = str(np.round(np.min(matrix[WLC, exposure,:,:]), rounding))
    return minimum_in_matrix

# This is currently not setting the masked values (which might make a square) to some very negative value... Use something else 
def find_brightestPoint_outsideCenter(image, radius): # define a f, find brightest point outside center w/ input radius, in an imshow matrix
    # Get image dimensions
    rows, cols = image.shape
    print('\nrows and cols are:', rows, cols) 

    # Skip center point
    center_x, center_y = int(cols / 2), int(rows / 2)
    print('\ncenter_x and center_y are:', center_x, center_y) 

    # Define search area based on radius
    min_x = max(0, center_x - radius)
    max_x = min(cols, center_x + radius)
    min_y = max(0, center_y - radius)
    max_y = min(rows, center_y + radius)
    print('\nmin_x, max_x, min_y and max_y are:', min_x, max_x, min_y, max_y) 

    # Extract sub-image excluding center
    search_area = image[0:cols, 0:rows] #[min_y:max_y, min_x:max_x]
    original_SA_shape = search_area.shape # Maybe this goes after the masking... ... ...
    print('\ninitially, np.shape(search_area) is:', np.shape(search_area)) 
    plt.imshow(search_area, cmap=plt.cm.viridis, alpha=0.9, interpolation='bilinear')
    plt.title('This is the 1st search area, so the full 200x200 matrix') 
    plt.xlim(0, rows)
    plt.ylim(0, cols)
    plt.colorbar()
    plt.show()

    # Mask out center to avoid including it in search (I think this should select the min_x...max_y pixels, not the center-min...center+max) 
    mask = np.ones(search_area.shape, dtype=bool)
    #mask[center_y - min_y : center_y - min_y + 1, center_x - min_x : center_x - min_x + 1] = False
    #mask[center_y - min_y : center_y - max_y, center_x - min_x : center_x - max_x] = -1000 #False
    mask[min_y : max_y, min_x : max_x] = -1000 #False
    search_area = search_area[mask]
    print('\nafter masking (before unravel), np.shape(search_area) (which is the same as search_area.shape) is:', np.shape(search_area))

    # Find brightest point (index) and intensity
    brightest_index = np.argmax(search_area)
    brightest_value = search_area.flat[brightest_index]
    print('\nbrightest_index and brightest_value are:', brightest_index, brightest_value) 

    # Convert index back to coordinates relative to search area
    y_offset, x_offset = np.unravel_index(brightest_index, original_SA_shape)
    search_area = np.reshape(search_area, original_SA_shape) # unravel_index
    print('\ny_offset and x_offset (which seem to be the coordinates of the brightest pixel!) are:', y_offset, x_offset)
    print('\nafter masking central values should be very negative or DNE, some are:', np.round(search_area[100,100],2), np.round(search_area[91,100],2), np.round(search_area[109,100],2), np.round(search_area[100,91],2), np.round(search_area[100,109]),2)
    plt.imshow(search_area, cmap=plt.cm.viridis, alpha=0.9, interpolation='bilinear') # delete after this ASAP: #, extent=extent)
    plt.title('This is the 2nd search area, so without the central ([85:115]) range') 
    plt.xlim(0, rows)
    plt.ylim(0, cols)
    plt.colorbar()
    plt.show()
    print('END FUNCTION\n\n\n') 

    # Return coordinates relative to entire image and intensity
    return [min_x + x_offset, min_y + y_offset], brightest_value

def make_image_center_unsearched(image, radius_threshold):
    rows, cols = image.shape
    center_x, center_y = int(cols / 2), int(rows / 2)
    #print('\ninitially, rows, cols, center_x and center_y are:', rows, cols, center_x, center_y)
    #plt.imshow(image, cmap=plt.cm.viridis, interpolation='bilinear') # alpha=0.9, 
    #plt.title('This is the original image, so the full 200x200 matrix') 
    #plt.colorbar()
    #plt.show()
    for i in range(len(image[:,0])):
        for j in range(len(image[0,:])):
            radius = np.sqrt( (i-center_x)**2 + (j-center_y)**2 )
            if radius < radius_threshold:
                image[i,j] = False #-1000
    #print('\nafter masking central values should be 0 or DNE, some are:', np.round(image[100,100],12), np.round(image[91,100],12), np.round(image[109,100],12), np.round(image[100,91],12), np.round(image[100,109],12) )
    brightest_index = np.argmax(image)
    brightest_value = image.flat[brightest_index] 
    y_bright, x_bright = np.unravel_index(brightest_index, image.shape) #original_SA_shape
    #print('\nbrightest_index (unraveled) and brightest_value are:', brightest_index, brightest_value) 
    #print('\ny_bright and x_bright (the coordinates of the brightest pixel!) are:', y_bright, x_bright)
    #plt.figure(figsize = (20,20))
    #plt.imshow(image, cmap=plt.cm.viridis, interpolation='bilinear')
    #plt.title('This is the 2nd image, so the middle r='+str(radius_threshold)+' is blank') 
    #plt.xlabel('x'), plt.ylabel('y') 
    #plt.colorbar()
    #plt.show()
    return brightest_value, y_bright, x_bright # image
    
def dimmest_pixel_center_unsearched(image, radius_threshold): 
    rows, cols = image.shape
    center_x, center_y = int(cols / 2), int(rows / 2)
    for i in range(len(image[:,0])):
        for j in range(len(image[0,:])):
            radius = np.sqrt( (i-center_x)**2 + (j-center_y)**2 )
            if radius < radius_threshold:
                image[i, j] = False
    dimmest_index = np.argmin(image)
    dimmest_value = image.flat[dimmest_index] 
    y_dim, x_dim = np.unravel_index(dimmest_index, image.shape)
    return dimmest_value, y_dim, x_dim
        
    
        
    

###################################################################################################



@typechecked
def postprocessor(images: np.ndarray,
                  angles: np.ndarray,
                  scales: Optional[np.ndarray],
                  pca_number: Union[int, Tuple[Union[int, np.int32, np.int64], Union[int, np.int32, np.int64]]],
                  pca_sklearn: Optional[PCA] = None,
                  im_shape: Union[None, tuple] = None,
                  indices: Optional[np.ndarray] = None,
                  mask: Optional[np.ndarray] = None,
                  processing_type: str = 'ADI'):

    """
    Function to apply different kind of post processings. It is equivalent to
    :func:`~pynpoint.util.psf.pca_psf_subtraction` if ``processing_type='ADI'` and
    ``mask=None``.

    Parameters
    ----------
    images : np.array
        Input images which should be reduced.
    angles : np.ndarray
        Derotation angles (deg).
    scales : np.array
        Scaling factors
    pca_number : tuple(int, int)
        Number of principal components used for the PSF subtraction.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the basis if not set to None.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if ``pca_sklearn`` is not set to None.
    indices : np.ndarray, None
        Non-masked image indices. All pixels are used if set to None.
    mask : np.ndarray
        Mask (2D).
    processing_type : str
        Post-processing type:
            - ADI: Angular differential imaging.
            - SDI: Spectral differential imaging.
            - SDI+ADI: Spectral and angular differential imaging.
            - ADI+SDI: Angular and spectral differential imaging.

    Returns
    -------
    np.ndarray
        Residuals of the PSF subtraction.
    np.ndarray
        Derotated residuals of the PSF subtraction.
    """

    if not isinstance(pca_number, tuple):
        pca_number = (pca_number, -1)

    if mask is None:
        mask = 1.

    res_raw = np.zeros(images.shape)
    res_rot = np.zeros(images.shape)

    if processing_type == 'ADI':
        if images.ndim == 2:
            res_raw, res_rot = pca_psf_subtraction(images=images*mask,
                                                   angles=angles,
                                                   scales=None,
                                                   pca_number=pca_number[0],
                                                   pca_sklearn=pca_sklearn,
                                                   im_shape=im_shape,
                                                   indices=indices)

        elif images.ndim == 4:
            for i in range(images.shape[0]):
                res_raw[i, ], res_rot[i, ] = pca_psf_subtraction(images=images[i, ]*mask,
                                                                 angles=angles,
                                                                 scales=None,
                                                                 pca_number=pca_number[0],
                                                                 pca_sklearn=pca_sklearn,
                                                                 im_shape=im_shape,
                                                                 indices=indices)

    elif processing_type == 'SDI':
        for i in range(images.shape[1]):
            im_scaled = sdi_scaling(images[:, i, :, :], scales)

            res_raw[:, i], res_rot[:, i] = pca_psf_subtraction(images=im_scaled*mask,
                                                               angles=np.full(scales.size,
                                                                              angles[i]),
                                                               scales=scales,
                                                               pca_number=pca_number[0],
                                                               pca_sklearn=pca_sklearn,
                                                               im_shape=im_shape,
                                                               indices=indices)

    elif processing_type == 'SDI+ADI':
        # SDI
        res_raw_int = np.zeros(res_raw.shape)

        for i in range(images.shape[1]):
            im_scaled = sdi_scaling(images[:, i], scales)

            res_raw_int[:, i], _ = pca_psf_subtraction(images=im_scaled*mask,
                                                       angles=None,
                                                       scales=scales,
                                                       pca_number=pca_number[0],
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)

        # ADI
        for i in range(images.shape[0]):
            res_raw[i], res_rot[i] = pca_psf_subtraction(images=res_raw_int[i]*mask,
                                                         angles=angles,
                                                         scales=None,
                                                         pca_number=pca_number[1],
                                                         pca_sklearn=pca_sklearn,
                                                         im_shape=im_shape,
                                                         indices=indices)

    elif processing_type == 'ADI+SDI':
        # ADI
        res_raw_int = np.zeros(res_raw.shape)

        for i in range(images.shape[0]):
            res_raw_int[i], _ = pca_psf_subtraction(images=images[i, ]*mask,
                                                    angles=None,
                                                    scales=None,
                                                    pca_number=pca_number[0],
                                                    pca_sklearn=pca_sklearn,
                                                    im_shape=im_shape,
                                                    indices=indices)

        # SDI
        for i in range(images.shape[1]):
            im_scaled = sdi_scaling(res_raw_int[:, i], scales)

            res_raw[:, i], res_rot[:, i] = pca_psf_subtraction(images=im_scaled*mask,
                                                               angles=np.full(scales.size,
                                                                              angles[i]),
                                                               scales=scales,
                                                               pca_number=pca_number[1],
                                                               pca_sklearn=pca_sklearn,
                                                               im_shape=im_shape,
                                                               indices=indices)

    elif processing_type == 'CODI':
        # flatten images from 4D to 3D
        ims = images.shape
        im_scaled_flat = np.zeros((ims[0]*ims[1], ims[2], ims[3]))
        scales_flat = np.zeros((ims[0]*ims[1]))
        angles_flat = np.zeros((ims[0]*ims[1]))
        for i in range(ims[1]):
            im_scaled_flat[i*ims[0]:(i+1)*ims[0]] = sdi_scaling(images[:, i], scales)
            scales_flat[i*ims[0]:(i+1)*ims[0]] = scales
            angles_flat[i*ims[0]:(i+1)*ims[0]] = angles[i]

        # codi
        res_raw_flat, res_rot_flat = pca_psf_subtraction(images=im_scaled_flat*mask,
                                                         angles=angles_flat,
                                                         scales=scales_flat,
                                                         pca_number=pca_number[0],
                                                         pca_sklearn=pca_sklearn,
                                                         im_shape=im_shape,
                                                         indices=indices)

        # inflate images from 3D to 4D
        for i in range(ims[1]):
            res_raw[:, i] = res_raw_flat[i*ims[0]:(i+1)*ims[0]]
            res_rot[:, i] = res_rot_flat[i*ims[0]:(i+1)*ims[0]]
            
            
            

    
    ############################################################################################################### 
    ### ### ### Make plots of original res_raw images (WLC = 8, for 32 exposures), and mitigated images ### ### ###
    ###############################################################################################################
    
    # Define variables: An original res_raw that preserves the values passed before my edits, and a new matrix for testing (negatives)
    # It seems that the simpler y = x, without np.multiply leads to some weird equivalences and doesn't generate clean new variables
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    res_raw_original = np.multiply(res_raw, 1) 
    res_raw_neg  = np.multiply(res_raw, -1)  # THIS IS IMPORTANT... AFFECTS FINAL OUTCOME, WHICH IT SHOULDN'T - 10 SEP 2023
    res_raw_10mb = np.multiply(res_raw,  1)
    res_raw_20mb = np.multiply(res_raw,  1)
    res_raw_30mb = np.multiply(res_raw,  1)
    
    res_rot_original = np.multiply(res_rot, 1) 
    res_rot_neg  = np.multiply(res_rot, -1)  # THIS IS IMPORTANT... AFFECTS FINAL OUTCOME, WHICH IT SHOULDN'T - 10 SEP 2023
    res_rot_10mb = np.multiply(res_rot,  1)
    res_rot_20mb = np.multiply(res_rot,  1)
    res_rot_30mb = np.multiply(res_rot,  1)

    # Determine data characteristics before modifications
    shapeResRaw, shapeResRot = np.shape(res_raw), np.shape(res_rot)
    print("\n\n\tbefore modification, np.shape(res_raw):", shapeResRaw) #(39, 32, 200, 200)
    print("\tbefore modification, np.shape(res_rot):", shapeResRot)

    # demonstration of feasibility--this runs! Now check with plots, and then find brightest pixel in all 32 exposures
    image = res_raw[0,0,:,:]
    radius_threshold = 13
    print("\tnp.shape(image):", np.shape(image))
    #coordinates, intensity = find_brightestPoint_outsideCenter(image, 15) # 15 is a good first guess for the 51 Eri b mock FPI 
    #print(f'Brightest point outside center (radius 50): {coordinates} with intensity {intensity}')
    brightest_value, y_bright, x_bright = make_image_center_unsearched(image, radius_threshold)

    # Use the above function to find the brightest and darkest (non-speckle) pixels in each of the 32 pixels
    exposures_num = len(res_raw[0,:,0,0]) #32
    pixel_brightest_mag = np.zeros(exposures_num) 
    pixel_brightest_x = np.zeros(exposures_num)
    pixel_brightest_y = np.zeros(exposures_num)
    pixel_dimmest_mag = np.zeros(exposures_num) 
    pixel_dimmest_x = np.zeros(exposures_num)
    pixel_dimmest_y = np.zeros(exposures_num)
    for i in range(exposures_num):
        image = res_raw[1,i,:,:]
        brightest_value, y_bright, x_bright = make_image_center_unsearched(image, radius_threshold)
        pixel_brightest_mag[i] = brightest_value
        pixel_brightest_x[i] = x_bright
        pixel_brightest_y[i] = y_bright
        
        dimmest_value, y_dim, x_dim = dimmest_pixel_center_unsearched(image, radius_threshold)
        pixel_dimmest_mag[i] = dimmest_value
        pixel_dimmest_x[i] = x_dim
        pixel_dimmest_y[i] = y_dim
        

    # Plot the magnitudes of the birghtest pixel on exposure number
    plt.plot(np.add(range(exposures_num),1), pixel_brightest_mag, 'o')
    plt.xlabel('Exposure number'), plt.ylabel('Magnitude (brightness)')
    plt.title('Brightest pixel magnitude on exposure number') 
    plt.show()
    plt.plot(np.add(range(exposures_num),1), pixel_dimmest_mag, 'o')
    plt.xlabel('Exposure number'), plt.ylabel('Magnitude (brightness)')
    plt.title('Dimmest pixel magnitude on exposure number') 
    plt.show()
    brightest_bright_idx = np.argmax(pixel_brightest_mag) 
    brightest_bright_mag = np.max(pixel_brightest_mag)
    dimmest_dim_idx = np.argmin(pixel_dimmest_mag) 
    dimmest_dim_mag = np.min(pixel_dimmest_mag)
    print('The brightest pixel of all exposures\' is #'+str(brightest_bright_idx+1)+', and magnitude of:'+str(np.round(brightest_bright_mag, 4))) 
    print('it is at coordinates: ['+str(pixel_brightest_x[brightest_bright_idx])+', '+str(pixel_brightest_y[brightest_bright_idx])+'].') 
    print('Use these for the center of the CB\'s neutral or dimmest point in the addition below!')
    print('#####################################################################################')
    print('The dimmest pixel of all exposures\' is #'+str(dimmest_dim_idx+1)+', and magnitude of:'+str(np.round(dimmest_dim_mag, 4))) 
    print('it is at coordinates: ['+str(pixel_dimmest_x[dimmest_dim_idx])+', '+str(pixel_dimmest_x[dimmest_dim_idx])+'].') 
    print('Use these for the center of the CB\'s brightest point to add below!')
    cmap = plt.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, num=32)) 
    for i in range(len(pixel_brightest_x)):
        size = -(6 / 32)*i + (6 + 1) # where 6(+1) is the maximum size, 32 is the number of exposures, and 1 is the minimum size 
        if i == brightest_bright_idx:
            size = size*3
        plot_max = plt.plot(pixel_brightest_x[i], pixel_brightest_y[i], 'o',  c=colors[i], markersize=size, label=str(i+1)) #  cmap=cmap, #scatter
    #plt.colorbar(plot_max, label='exposure #')
    plt.ylim([np.min(pixel_brightest_y)-6, np.max(pixel_brightest_y)+6])
    plt.xlabel('X'), plt.ylabel('Y')
    plt.title('Locations of Maximum Signal outside of central\nspeckle noise (over 32 exposures in color gradient)')
    plt.legend(ncol=2, bbox_to_anchor=(1.4, 1)) # Shows the exposure number (by color gradient) # (1.4=x_dist, 1.1=y_dist)
    plt.show() 
    cmap = plt.colormaps['plasma']
    colors_dim = cmap(np.linspace(0, 1, num=32)) 
    for i in range(len(pixel_brightest_x)):
        size = -(6 / 32)*i + (6 + 1) # where 6(+1) is the maximum size, 32 is the number of exposures, and 1 is the minimum size 
        if i == brightest_bright_idx:
            size = size*3
        plot_max = plt.plot(pixel_dimmest_x[i], pixel_dimmest_y[i], 'o',  c=colors_dim[i], markersize=size, label=str(i+1))
    plt.ylim([np.min(pixel_dimmest_y)-6, np.max(pixel_dimmest_y)+6])
    plt.xlabel('X'), plt.ylabel('Y')
    plt.title('Locations of _Minimum_ Signal outside of central\nspeckle noise (over 32 exposures in color gradient)')
    plt.legend(ncol=2, bbox_to_anchor=(1.4, 1)) 
    plt.show() 

    # By visual inspection, determine how many periods go through the oscillation (1.6?) over the 32 exposures, & how many pixels are in CB squares
    '''
    for i in [1, 10, 20, 30]:
        print('\n\n\nnow plotting the WLC of:', i)
        for j in range(exposures_num):
            image = res_raw[i,j,:,:]
            plt.imshow(image, origin='lower', cmap='Blues_r')
            plt.title('The WLC: '+str(i)+' and Exposure #:'+str(j))
            plt.show()
    '''
    # when WLC =  1, the 32 exposures oscillate over slightly more than 1/2 of a period
    # when WLC = 10, the 32 exposures oscillate over about 1/2 of a period (maybe a little more) 
    # when WLC = 20, the 32 exposures oscillate within the speckle noise, appearing breifly in the middle 
    # when WLC = 30, the 32 exposures oscillate over about 1/2 of a period 
    
    # Define the radians_in_exposure_oscillation variable below to be the periodicity of the exposures above
    radians_in_exposure_oscillation = 3.3 # slightly more than pi radians (1/2 a period) 
    wavelength_channels = [ 954.32282227,  962.62163342,  971.8455533, 981.89907773,  991.62082995, 1001.70638262,
                        1012.66201578, 1023.29215927, 1033.63748036, 1043.90899136, 1054.17332605, 1064.93904728,
                        1075.51906673, 1085.48399237, 1095.372864, 1105.07626751, 1115.18984928, 1127.09242581,
                        1138.79673007, 1149.43864103, 1159.42615783, 1169.31067405, 1179.38222723, 1189.4464981,
                        1199.5333784, 1209.88668205, 1220.04015567, 1229.96503806, 1240.57338913, 1250.6362063,
                        1260.9653672,  1271.4195118,  1281.94965148,1291.7051642,  1300.62445043, 1309.61923874,
                        1318.81874937, 1327.10100597, 1333.64664519 ] # In nanometers
    WLC_num = 39 # len(wavelength_channels)
    angles_BPic20 = [-17.10792503, -18.04861444, -19.00754622, -19.94279229, -20.87967121, -21.83354603, -22.80021385, 
                -23.7361827,  -24.67226941, -25.64268633, -26.66322205, -27.64449798, -28.60997872, -29.57157742, 
                -30.54446954, -31.52492384, -32.49536969, -33.46763605, -34.46496023, -35.42006557, -36.40736831, 
                -37.39081194, -38.40896634, -39.38236154, -40.37439087, -41.34595291, -42.32240888, -43.28713799, 
                -44.25890071, -45.2366961, -46.19115248, -47.14049291]
    # Given a range (1.5*pi radians), make a discrete sign function that is evaluated in the spacing that the wavelength channels are distributed
    wlMin, wlMax, wlDistribution, wlDistrib_sin = np.min(wavelength_channels), np.max(wavelength_channels), [], []
    for i in range(WLC_num):
        wlDistribution.append(radians_in_exposure_oscillation*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) # the 7.6 is ___________? 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
    

    ###################################################################################################################
    ### Determine optimal variables by inspection: Diameter/radius, approximately maximum brightness, maybe more... ###
    ###################################################################################################################
    # Plot some images of 51 Eri b analogs with CBs of varying size (D) parallel/next to them
    diameter = [8]
    x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, Xdelay, Ydelay, decay_factor = 5, 5, [3, 3], 18, 1, 3.14, 0, 0.4
    location = [68, 93] # [y, x], y:150-->100 from floor to middle. x:40-->80 from left to right and a little up!
    for D in diameter:
        for i in range(WLC_num):
            checkerboard, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, Xdelay, Ydelay, decay_factor) # wlD,
            for j in range(len(angles_BPic20)):
                aligned_CB, rotMat = align_CB_to_angle(checkerboard, 70) 
                mitigated_signal = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB)
                if i == 1 and j == brightest_bright_idx:
                    print('enter if statement. j, the brightest_bright_idx, is:', j, 'plot the original and mitigated data')
                    mitigated_signal_best, D_best, i_best = mitigated_signal, D, i
                    '''
                    # Plot the original and mitigated signals with the 1 colorbar
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                    fig.suptitle('FPI Original Data and Checkerboard Artifact-Mitigation', fontsize=14)
                    axes[0].imshow(res_raw[i,j,:,:], origin='lower', cmap='Blues_r', vmin=dimmest_dim_mag, vmax=brightest_bright_mag)
                    axes[0].set_title('Original data (brightest pixel WLC = '+str(i+1)+' & exp = '+str(j+1)+')')
                    axes[1].imshow(mitigated_signal, origin='lower', cmap='Blues_r', vmin=dimmest_dim_mag, vmax=brightest_bright_mag)
                    axes[1].set_title('Mitigated (D = '+str(D)+', bright pixel WLC = '+str(i+1)+' & exp = '+str(j+1)+')')
                    sm = plt.colorbar(axes=[axes[0], axes[1]], label='Colorbar')  # Colorbar for both axes
                    sm.set_clim(dimmest_dim_mag, brightest_bright_mag)
                    #fig.colorbar(im0, ax=[ax1, ax2], label='Colorbar')                    #plt.tight_layout()
                    plt.show()
                    #ax[1].show()
                    '''
        #Same as above, but outside the for loop
        plt.imshow(res_raw[1,brightest_bright_idx,:,:], origin='lower', cmap='Blues_r', vmin=dimmest_dim_mag, vmax=brightest_bright_mag)
        plt.title('FPI Original data; brightest pixel WLC='+str(i_best+1)+' & exp='+str(brightest_bright_idx+1)+'.'), plt.colorbar(), 
        plt.xlabel('X or RA'), plt.ylabel('Y or Dec'), plt.show()
        plt.imshow(mitigated_signal_best, origin='lower', cmap='Blues_r', vmin=dimmest_dim_mag, vmax=brightest_bright_mag)
        plt.title('FPI, CB mit, D = '+str(D_best)+'; WLC='+str(i_best+1)+' & exp='+str(brightest_bright_idx+1)+'.'), plt.colorbar(), 
        plt.xlabel('X or RA'), plt.ylabel('Y or Dec'), plt.show()
        
    print('It seems like a diameter = 8 pixels, location = [68, 93], & rotation = 70 is optimal... ')
    # The CB maxima should be the positive of the artifact minimum...


    
    #########################################################################################################################
    ### Run a nested for loop to find the best exoplanet signal (that can be transfered to any other potential exoplanet) ###
    #########################################################################################################################
    print('Optimizing the checkerboard... ... ...')
    diameter = [8];                                D = diameter
    radius = [int(D[0]/2)];                        R = radius
    separation = 50;                               S = separation
    decay_factor = [0.3, 0.4, 0.5];                DF= decay_factor
    #angular_squares, radial_squares = 4, 4 # If completes, delete line. I think there's no code that uses this now - 5 July 23
    max_brightness = [15, 18, 21];                 MB= max_brightness # ~20!  half_edge = np.sqrt(radius**2 + separation**2) - separation + radius
    amp_nonRad_less = [1, 0.8];                    AL= amp_nonRad_less
    x_squares, y_squares = 5, 5
    center_square = [3, 3]
    location_x = [68] # maybe switch x & y 
    location_y = [93] #location, location_demo = (44, 88), (44, 125)
    Xdelay = [0.5, 1.5, 2.5]
    y_delay = [0, 2]
    wlD = wlDistribution[i]
    #amp_radial_boost = 1 
    mitigated_signal = np.zeros([39, 32, 200, 200])
    minimums_increase = np.zeros([39, 32]) 
    negInt_increase = np.zeros([39, 32])
    negInt_inc_alt = np.zeros([39, 32])

    for_loop_counter = 0
    for i in diameter:
        break # Remove this to run the code
        for j in max_brightness:
            for k in amp_nonRad_less:
                for l in location_x:
                    for m in location_y:
                        for n in Xdelay:
                            for o in y_delay: # This used to be wlDistribution: #wlD:, which was (probs) useful when Xdelay didn't cover a period 
                                for p in decay_factor: 
                                    # Calculate the CB modification
                                    #print('i, x_squares, y_squares, center_square, j, k, [l, m], n, o, p', i, x_squares, y_squares, center_square, j, k, [l, m], n, o, p) 
                                    checkerboard, CB4D = make_oscillating_checkerboard_noGap(i, x_squares, y_squares, center_square, j, k, location, n, o, p)
                                    #(diameter, x_sq, y_sq, cent_sqr, max_bright, amp_nonRad_less, loc, Xdelay, wlDistrib, decay_factor)# wlD
                                    for q in range(WLC_num):
                                        for r in range(len(angles_BPic20)): 
                                            aligned_CB, rotMat = align_CB_to_angle(checkerboard, 70) # -1*angles_BPic20[r]
                                            mitigated_signal[q,r,:,:] = add_wavy_checkerboard(res_raw[q,r,:,:], aligned_CB)
                                            # Charaterize the mitigation (minimum's increase, maximum's preservation, (neg integral's increase)
                                            minimums_increase[q, r] = np.min(res_raw[q,r,:,:]) - np.min(mitigated_signal)
                                            mask_orig = res_raw[q,r,:,:] < 0
                                            mask_miti = mitigated_signal < 0
                                            negInt_increase[q, r] = np.sum(res_raw[q,r,:,:][mask_orig]) - np.sum(mitigated_signal[mask_miti])
                                            negInt_inc_alt[q, r] = np.sum(res_raw[q,r,:,:][res_raw[q,r,:,:] < 0]) - np.sum(mitigated_signal[mitigated_signal < 0])
                                            for_loop_counter += 1
                                            if for_loop_counter%1000 == 0:
                                                print('The for_loop_counter has reached', for_loop_counter) 
                                                print('\tmax_brightness\'s in test and currently analyzing diameter:', max_brightness, j) 
                                                
                                    # Determine the best mitigations, show a plot of them
    print('The for_loop_counter has increased from 0 to', for_loop_counter)
    print('The largest increase in minimum value (np.argmax(minimums_increase)) is at WLC and Exposure:', np.argmax(minimums_increase))
    print('The largest increase in the image\'s <0 integral () is at WLC and Exposure:', np.argmax(negInt_increase))
    print('The largest increase in the image\'s <0 integral () is at WLC and Exposure:', np.argmax(negInt_inc_alt))
    print('The _____ and _____ should be the same, and are:', negInt_increase, negInt_inc_alt)
    print('\tTheir subtraction should be 0, and is:', np.subtract(negInt_increase, negInt_inc_alt))


    ################################################################################################
    # Set-up, define, or use CheckerBoard parameters
    diameter = 5;                    D = diameter
    radius = int(diameter/2);        R = radius
    separation = 50;                 S = separation
    decay_factor = 0.4;              DF= decay_factor
    angular_squares, radial_squares = 4, 4 # I think there's no code that uses this now - 5 July 23
    max_brightness = 20;             MB= max_brightness # ~20!  half_edge = np.sqrt(radius**2 + separation**2) - separation + radius
    amp_nonRad_less = 1;             AL= amp_nonRad_less
    x_squares, y_squares, center_square, location, location_demo = 5, 5, [3, 3], (44, 88), (44, 125)
    amp_radial_boost = 1 # MB=24,18,10 --> too bright

    x_squares, y_squares, center_square, location, location_demo, Xdelay = 5, 5, [3, 3], (44, 88), (44, 125), diameter / 4
    location = (int(pixel_brightest_y[brightest_bright_idx] - (17)), int(pixel_brightest_x[brightest_bright_idx])) # From the maximum brightests' finding code above
    print('location should be integers (~88, ~88) and is', location)
    MB_10, MB_20, MB_30 = 10, 40, 90
    working_count = 0
    #fig, axs = plt.subplots(nrows=2, ncols=WLC_num, figsize=(8*WLC_num, 14))# sharey=False)
    
    
    
    for i in range(WLC_num):
        #radians_in_exposure_oscillation = 3.5 (it used to be ~10 for the beta pic FPI)   
        wlDistribution.append(radians_in_exposure_oscillation*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) # This will replace 39 WLCs with 1 period, which still needs to move ~10 radians so 10rad/1wlDistribution
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        #checkerboard_demo, CB4D_D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location_demo, Xdelay, wlD, decay_factor) # Ydelay2,
        #checkerboard_Ydelay, CB4D_Ydelay = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, Xdelay, Ydelay, decay_factor)
        checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, Xdelay, wlD, decay_factor) # Ydelay,
        CB_10, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB_10, AL, location, Xdelay, wlD, DF)
        CB_20, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB_20, AL, location, Xdelay, wlD, DF)
        CB_30, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB_30, AL, location, Xdelay, wlD, DF)
        #mitigated_signal_D = add_wavy_checkerboard(signalMatrix_FPI, checkerboard_demo)
        #mitigated_signal_Ydelay = add_wavy_checkerboard(signalMatrix_FPI, checkerboard_Ydelay)
        
        ### ### Align the CB, and then add it to the res_raw
        #print(np.shape(res_raw))  # (39, 32, 200, 200)!
        for j in range(len(angles_BPic20)):
            #print("\nj:"+str(j)+". -(j+1):"+str(-(j+1))+"\nangles_BPic20[j]:"+str(angles_BPic20[j])+"\nangles_BPic20[-j]"+str(angles_BPic20[-(j+1)]))
            #print("-1*angles_BPic20[j] should start with: -17.1, and is: "+str(-1*angles_BPic20[j]))     # yes... the angles are going in reverse order... 
            #print("-1*angles_BPic20[-(j+1)] start with: -47.1, and is: "+str(-1*angles_BPic20[-(j+1)]))
            # Test: instead of angles = 60 and -60, try 60 and 300 or something positive... 10 Sep 2023
            aligned_CB, rotMat = align_CB_to_angle(checkerboard, -1*angles_BPic20[j])                   # a small-ish 50,50 matrix
            aligned_CB_10, rotMat_10 = align_CB_to_angle(CB_10, -1*angles_BPic20[j])                   # a small-ish 50,50 matrix
            aligned_CB_20, rotMat_20 = align_CB_to_angle(CB_20, -1*angles_BPic20[j])                   # a small-ish 50,50 matrix
            aligned_CB_30, rotMat_30 = align_CB_to_angle(CB_30, -1*angles_BPic20[j])                   # a small-ish 50,50 matrix
            aligned_CB_neg, rotMat_neg = align_CB_to_angle(checkerboard, -1*angles_BPic20[-(j+1)])      # a small-ish 50,50 matrix
            
            #     THESE WORKED! Separated by a set 120 degree difference
            #print(np.shape(aligned_CB))
            #print(np.shape(aligned_CB_neg))
            #fig, axs = plt.subplots(nrows=1, ncols=2)#, figsize=(32*6, 32*0.625))
            #axs[0].title.set_text('Testing the alignment of CBs in plots, j (exposure) is: '+str(j))
            #axs[0].imshow(aligned_CB, origin='lower', cmap='Blues_r')
            #axs[1].imshow(aligned_CB_neg, origin='lower', cmap='Blues_r')
            #plt.show()
            
        
            mitigated_signal = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB)                # a 200,200 image
            mitigated_signal_10 = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_10)          # a 200,200 image
            mitigated_signal_20 = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_20)          # a 200,200 image
            mitigated_signal_30 = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_30)          # a 200,200 image
            
            mitigated_signal_neg = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_neg)  # a 200,200 image
            #mitigated_signal = add_wavy_checkerboard(np.zeros((200,200)), aligned_CB)          # The CB on a blank background
            #mitigated_signal_neg = add_wavy_checkerboard(np.zeros((200,200)), aligned_CB_neg)  # The negative CB on a blank
            
            '''     THESE WORKED! Separated by an 120 degree input            
            print(np.shape(mitigated_signal))
            print(np.shape(mitigated_signal_neg))
            fig, axs = plt.subplots(nrows=1, ncols=2)#, figsize=(32*6, 32*0.625))
            axs[0].title.set_text('Testing the mitigated signals in plots, j (exposure) is: '+str(j+1))
            axs[0].imshow(mitigated_signal, origin='lower', cmap='Blues_r')
            axs[1].imshow(mitigated_signal_neg, origin='lower', cmap='Blues_r')
            plt.show()
            '''
        
            '''     print shape of res raw,,, should be 200x200...?  ##### NEXT STEP HERE 8 SEP 2023!!! ###
            print('shape of res_raw[ij]', np.shape(res_raw[i,j,:,:]))           # 200x200
            print('shape of res_raw_neg[ij]', np.shape(res_raw_neg[i,j,:,:]))   # 200x200
            print('shape of mit_sig', np.shape(mitigated_signal))               # 200x200
            print('shape of mit_sig_neg', np.shape(mitigated_signal_neg))       # 200x200
            '''
            
            
            '''# Before: Print res_raw[i,j,:,:], mitigated_signal, res_raw_neg[i,j,:,:], and mitigated_signal_neg 
            if j == 21 and i == 21: # used: % 21 ...
                print('\n\n\tFIRST...\n The WLC and Exposure are:', i, j)
                print('res_raw[i,j,73:84,57:69]', res_raw[i,j,76:82,58:68])
                print('mitigated_signal[73:84,57:69]', mitigated_signal[76:82,58:68])
                print('res_raw_neg[i,j,73:84,57:69]', res_raw_neg[i,j,76:82,58:68])
                print('mitigated_signal_neg[73:84,57:69]', mitigated_signal_neg[76:82,58:68])
            ''' 
                
            # Assign values in the res_raw matrix
            res_raw[i,j,:,:] = mitigated_signal
            res_raw_10mb[i,j,:,:] = mitigated_signal_10
            res_raw_20mb[i,j,:,:] = mitigated_signal_20
            res_raw_30mb[i,j,:,:] = mitigated_signal_30
            res_raw_neg[i,j,:,:] = mitigated_signal_neg # Check around here for errors! 8 Sep

            # After: Print res_raw[i,j,:,:], mitigated_signal, res_raw_neg[i,j,:,:], and mitigated_signal_neg 


            counter_mitSig = 0
            counter = 0
            high_threshold = 32
            high_count_treshold = 0

            if j == 21 and i == 21:
                print('Check the values of the reduction at this WLC and exposure, i, j =', i, j)            
                if len(mitigated_signal) != len(mitigated_signal_neg):
                    print('The matrices are different length... This is a problem to fix! (should I use shape instead?)') 
                    counter_mitSig = 1
                for k in range(200):
                    for l in range(200):
                        #if res_raw[i,j,k,l] != res_raw_neg[i,j,k,l]:
                        if mitigated_signal[k,l] != mitigated_signal_neg[k,l]:
                            counter_mitSig = 1
                if counter_mitSig == 1:
                    print('mitSig & its negative are different (this is GOOD)')
                else:
                    print('mitSig & its negative are the same, which is BAD!')
                
                if len(res_raw[i,j,:,:]) != len(res_raw_neg[i,j,:,:]):
                    print('The matrices are different length... This is a problem to fix! (should I use shape instead?)') 
                    counter = 1
                for k in range(200):
                    for l in range(200):
                        if res_raw[i,j,k,l] != res_raw_neg[i,j,k,l]:
                            counter = 1
                        if res_raw[i,j,k,l] > high_threshold: # In np.zeros case, >8 yields ~12 points; >16 yields 5 points. Use BIG for obs
                            print('High signal strength (>32) at coordinates k, l =', k, l) 
                            high_count_treshold += 1
                            if high_count_treshold >= 150:
                                print('The signal strength is over the threshold 150+ times! Execute a break command now') 
                                break
                if counter == 1:
                    print('resRaw & its negative are different (this is GOOD)')
                else:
                    print('resRaw & its negative are the same, which is BAD!')
            #working_count += 1
            #print("it's working, res_raw & res_raw_neg are different...")
        #print('\n\tThe working_count (number of differences between regular and negative rotation) is:', working_count)
        #equal_matrix_test(res_raw, res_raw_neg)
        
        '''mitigated_signal_unaligned = add_wavy_checkerboard(res_raw[i,:,:,:], checkerboard)
        for j in range(len(angles)):
            mitigated_signal = align_CB_to_angle(mitigated_signal_unaligned[j,:,:], angles[j]) #maybe -j
            mitigated_signal_backwards = align_CB_to_angle(mitigated_signal_unaligned[j,:,:], angles[-j])
            #should edit the last two dimensions so that they are printing coordinate vales in the region the CB has been applied
            #print('\ni = '+str(i)+'. Before the res_raw[i,8,99:101,99:101] =\n'+str(res_raw[i,8,99:101,99:101])+'.')
            res_raw[i,j,:,:] = mitigated_signal
            res_raw_backwards[i,j,:,:] = mitigated_signal_backwards
            #print('Now the res_raw[i,8,99:101,99:101] =\n'+str(res_raw[i,8,99:101,99:101])+'.')
        '''
        
        # Make 2 subplots... #1 is of res_raw, 1x4 of WLC=8, and exposures={4, 12, 20, 28}. #2 is of the mitigated

    # outside the for loop, plot the two subplots
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    fig.suptitle('Res_Raw over 4 different exposures')
    axs[0].imshow(res_raw[8, 4,:,:], origin='lower', cmap='Blues_r')
    axs[1].imshow(res_raw[8,12,:,:], origin='lower', cmap='Blues_r')
    axs[2].imshow(res_raw[8,20,:,:], origin='lower', cmap='Blues_r')
    axs[3].imshow(res_raw[8,28,:,:], origin='lower', cmap='Blues_r')
    plt.show()
    
    
    print('\tFinished running the \'for i in range(WLC_num):\' for loop!')
    
    print("\n\n\tbefore modification, np.shape(res_raw):", shapeResRaw)
    print("\tbefore modification, np.shape(res_rot):", shapeResRot)
    print("\n\tafter modification, np.shape(res_raw):", np.shape(res_raw))
    print("\tafter modification, np.shape(res_rot):", np.shape(res_rot))

    exposures = len(res_raw_original[0,:,0,0])
    print('The $exposures$ variable should be 32, and is:', exposures) 
    
    fig, axs = plt.subplots(nrows=5, ncols=exposures, figsize=(exposures*6, exposures*1.0))
    fig.suptitle('Res Raw plots - WLC #11 - varied CB maximum brightness') # or plt.suptitle('Res Raw plots')
    for i in range(exposures): # range(len(res_raw_original[0,:,0,0]))
        axs[0, i].title.set_text("Original - exp#"+str(i+1) + "max= " + maxF(res_raw_original,11,i,1) + ", min= " + minF(res_raw_original,11,i,1)) #+". min: "+str(np.round(np.min(mitigated_signal), 2)) +". max: "+str(np.round(np.max(mitigated_signal), 2)) )
        #axs[1, :].title.set_text("Modified_set_text") #    AttributeError: 'numpy.ndarray' object has no attribute 'title'
        #axs[1].set_title("Modified_set_title") #AttributeError: 'numpy.ndarray' object has no attribute 'set_title'
        axs[0, i].imshow(res_raw_original[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[1, i].title.set_text("Modified - exp#"+str(i+1) + ", max= " + maxF(res_raw,11,i,1) + ", min= " + minF(res_raw,11,i,1))
        axs[1, i].imshow(res_raw[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[2, i].title.set_text("10mb Modified - exp#"+str(i+1) + ", max= " + maxF(res_raw_10mb,11,i,1) + ", min= " + minF(res_raw_10mb,11,i,1))
        image10 = axs[2, i].imshow(res_raw_10mb[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[3, i].title.set_text("20mb Modified - exp#"+str(i+1) + ", max= " + maxF(res_raw_20mb,11,i,1) + ", min= " + minF(res_raw_20mb,11,i,1))
        image20 = axs[3, i].imshow(res_raw_20mb[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[4, i].title.set_text("30mb Modified - exp#"+str(i+1) + ", max= " + maxF(res_raw_30mb,11,i,1) + ", min= " + minF(res_raw_30mb,11,i,1))
        image30 = axs[4, i].imshow(res_raw_30mb[11, i,:,:], origin='lower', cmap='Blues_r')
        #axs[2, i].imshow(res_raw_neg[11, i,:,:], origin='lower', cmap='Blues_r')

        #IDEA: Collect the min's and max's for generating statistics and determining optimal exposure
        
        divider10 = make_axes_locatable(axs[2, i]) # i was 0 before, all colorbars stacked on left plots
        divider20 = make_axes_locatable(axs[3, i])
        divider30 = make_axes_locatable(axs[4, i])
        cax10 = divider10.append_axes('right', size='5%', pad=0.3) # size is colorbar (CB) thickness; pad is CB-IMShow separation
        cax20 = divider20.append_axes('right', size='8%', pad=0.3) 
        cax30 = divider30.append_axes('right', size='10%', pad=0.2) 
        #image1.set_clim(vmax=30) # vmin=0, COMMENT THIS LINE OUT TO SEE VARIETY WITHIN THE ALL-BLUE PLOTS
        cbar10= fig.colorbar(image10, cax=cax10, ticks=[np.min(res_raw_10mb[11, i,:,:]), 0, np.max(res_raw_10mb[11, i,:,:])])
        cbar20= fig.colorbar(image20, cax=cax20, ticks=[np.min(res_raw_20mb[11, i,:,:]), 0, np.max(res_raw_20mb[11, i,:,:])])
        cbar30= fig.colorbar(image30, cax=cax30, ticks=[np.min(res_raw_30mb[11, i,:,:]), 0, np.max(res_raw_30mb[11, i,:,:])])
        cbar10.set_label('Cbar10')        
        cbar20.set_label('Cbar20')
        cbar30.set_label('Cbar30')
        
        '''
        if i == 0:
            #image = ax[i].imshow(data[0][i], cmap='plasma', origin='lower', extent=[0.5, 0.5+int(len_PCAs[i]), 0.5, int(len_PCAs[i])+0.5], rasterized=True)
            divider=make_axes_locatable(axs[2, i]) # ax
            cax = divider.append_axes('right', size='10%', pad=0.2) # size is color-bar (CB) thickness, while pad is CB-HM separation\n",
            image.set_clim(vmax=30) # vmin=0, COMMENT THIS LINE OUT TO SEE VARIETY WITHIN THE ALL-BLUE PLOTS
            cbar= fig.colorbar(image10, cax=cax)
            cbar.set_label('Max brightness:', MB_10)
        ''' 
    
    #fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(36, 18))
    #axs[0].imshow(res_raw[8, 4,:,:], origin='lower', cmap='Blues_r')
    #axs[1].imshow(res_raw[8,12,:,:], origin='lower', cmap='Blues_r')
    #axs[2].imshow(res_raw[8,20,:,:], origin='lower', cmap='Blues_r')
    #axs[3].imshow(res_raw[8,28,:,:], origin='lower', cmap='Blues_r')
    
    # Save the figure
    print("Save raw Figure...")
    documents_path = "/home/dcpetit/Desktop/research_FPI/"
    filename_raw = "fpi_res_raw_figure.png" # or jpg
    full_path = documents_path + filename_raw #full_path = os.path.join(documents_path, filename)
    #plt.savefig(full_path) 
    #print(f"Figure saved to: {full_path}")
    
    plt.show()
    
    
    ###########################################################
    ### ### ### Above was res_raw; below is res_rot ### ### ###
    ###########################################################

    # I think the location variable must be changed now (constant?), as the location no longer rotates with each exposure... 3 Mar 2024
    #location   =  (44, 88)
    #location_i = [(44, 88), (44, 88), (44, 88), (44, 88), (44, 88), (44, 88), (44, 88), ...]

    # For each of the wavelengths...
    for i in range(WLC_num):
        # Define the phase of the sine function at this WLC
        wlDistribution.append(10*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        # Generate a checkerboard (at this phase, the location above, with the regular set of parameters) 
        checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, Xdelay, wlD, decay_factor)
        CB_10, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB_10, AL, location, Xdelay, wlD, DF)
        CB_20, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB_20, AL, location, Xdelay, wlD, DF)
        CB_30, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB_30, AL, location, Xdelay, wlD, DF)
        
        ### ### Align the CB, and then add it to the res_rot
        for j in range(len(angles_BPic20)):
            # Use a constant... try 3 Mar 2024 ... ... ... before had the above angles (not 0) from the FITS file to align, which was bad
            aligned_CB, rotMat = align_CB_to_angle(checkerboard, 0)                               # a small-ish 50,50 matrix
            aligned_CB_10, rotMat_10 = align_CB_to_angle(CB_10, 0)                            
            aligned_CB_20, rotMat_20 = align_CB_to_angle(CB_20, 0)                            
            aligned_CB_30, rotMat_30 = align_CB_to_angle(CB_30, 0)                            
            aligned_CB_neg, rotMat_neg = align_CB_to_angle(checkerboard, 0)                   
            
            
            mitigated_signal = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB)                # a 200,200 image
            mitigated_signal_10 = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB_10)
            mitigated_signal_20 = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB_20)
            mitigated_signal_30 = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB_30)
            
            mitigated_signal_neg = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB_neg)  # a 200,200 image
            
            # WARNING: THESE MIGHT BE CHANGED FROM THE RES_RAW SECTION... CAN I REUSE THESE?
            # ANSWER: YES, THINK IT'S GOOD NOW
            # Assign values in the res_rot matrix
            res_rot[i,j,:,:] = mitigated_signal
            res_rot_10mb[i,j,:,:] = mitigated_signal_10
            res_rot_20mb[i,j,:,:] = mitigated_signal_20
            res_rot_30mb[i,j,:,:] = mitigated_signal_30
            res_rot_neg[i,j,:,:] = mitigated_signal_neg # Check around here for errors! 8 Sep (2023?) 

            # After: Print res_rot[i,j,:,:], mitigated_signal, res_rot_neg[i,j,:,:], and mitigated_signal_neg 


            counter_mitSig = 0
            counter = 0

            if j == 21 and i == 21:
                print('Check the values of the reduction at this WLC and exposure, i, j =', i, j)            
                if len(mitigated_signal) != len(mitigated_signal_neg):
                    print('The matrices are different length... This is a problem to fix! (should I use shape instead?)') 
                    counter_mitSig = 1
                for k in range(200):
                    for l in range(200):
                        #if res_rot[i,j,k,l] != res_rot_neg[i,j,k,l]:
                        if mitigated_signal[k,l] != mitigated_signal_neg[k,l]:
                            counter_mitSig = 1
                if counter_mitSig == 1:
                    print('mitSig & its negative are different (this is GOOD)')
                else:
                    print('mitSig & its negative are the same, which is BAD!')
                
                if len(res_rot[i,j,:,:]) != len(res_rot_neg[i,j,:,:]):
                    counter = 1
                for k in range(200):
                    for l in range(200):
                        if res_rot[i,j,k,l] != res_rot_neg[i,j,k,l]:
                            counter = 1
                        if res_rot[i,j,k,l] > 32: # In np.zeros case, >8 yields ~12 points; >16 yields 5 points. Use BIG for obs
                            print('res_rot[i,j,k,l] > 32 is happening! This means that (the CB is too bright?)')
                if counter == 1:
                    print('resRot & its negative are different (this is GOOD)')
                else:
                    print('resRot & its negative are the same, which is BAD!')
         
        # Make 2 subplots... 1 is of res_rot, 1x4 of WLC=8, and exposures={4, 12, 20, 28}. 2 is of the mitigated

    # outside the for loop, *plot* the res_rot over 4 different exposures
    '''
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    fig.suptitle('Res_Rot over 4 different exposures (#1)')
    axs[0].imshow(res_rot[8, 4,:,:], origin='lower', cmap='Blues_r')
    axs[1].imshow(res_rot[8,12,:,:], origin='lower', cmap='Blues_r')
    axs[2].imshow(res_rot[8,20,:,:], origin='lower', cmap='Blues_r')
    axs[3].imshow(res_rot[8,28,:,:], origin='lower', cmap='Blues_r')
    plt.show()
    '''
    
    print('\tFinished running the \'for i in range(WLC_num):\' for loop!')
    
    print("\n\n\tbefore modification, np.shape(res_raw):", shapeResRaw)
    print("\tbefore modification, np.shape(res_rot):", shapeResRot)
    print("\n\tafter modification, np.shape(res_raw):", np.shape(res_raw))
    print("\tafter modification, np.shape(res_rot):", np.shape(res_rot))
    
    
    fig, axs = plt.subplots(nrows=5, ncols=exposures, figsize=(exposures*6, exposures*1.0))
    fig.suptitle('Res Rot (rotated) plots - WLC #11 ([11, i,:,:]) - With varied CB maximum brightness') # or plt.suptitle('Res Raw plots')
    for i in range(exposures): # range(len(res_rot_original[0,:,0,0]))
        axs[0, i].title.set_text("Original - exposure #"+str(i+1) + ", max="+maxF(res_rot_original,11,i,1)+", min="+minF(res_rot_original,11,i,1))
        #axs[1, :].title.set_text("Modified_set_text") #    AttributeError: 'numpy.ndarray' object has no attribute 'title'
        #axs[1].set_title("Modified_set_title") #           AttributeError: 'numpy.ndarray' object has no attribute 'set_title'
        axs[0, i].imshow(res_rot_original[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[1, i].title.set_text("res_rot #"+str(i+1) + ", max=" + maxF(res_rot,11,i,1) + ", min=" + minF(res_rot,11,i,1))
        axs[1, i].imshow(res_rot[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[2, i].title.set_text("res_rot_10mb #"+str(i+1) + ", max=" + maxF(res_rot_10mb,11,i,1) + ", min=" + minF(res_rot_10mb,11,i,1))
        image10 = axs[2, i].imshow(res_rot_10mb[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[3, i].title.set_text("res_rot_20mb #"+str(i+1) + ", max=" + maxF(res_rot_20mb,11,i,1) + ", min=" + minF(res_rot_20mb,11,i,1))
        image20 = axs[3, i].imshow(res_rot_20mb[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[4, i].title.set_text("res_rot_30mb #"+str(i+1) + ", max=" + maxF(res_rot_30mb,11,i,1) + ", min=" + minF(res_rot_30mb,11,i,1))
        image30 = axs[4, i].imshow(res_rot_30mb[11, i,:,:], origin='lower', cmap='Blues_r')
        
        #axs[2, i].imshow(res_rot_neg[11, i,:,:], origin='lower', cmap='Blues_r')
        
        divider10 = make_axes_locatable(axs[2, i])
        divider20 = make_axes_locatable(axs[3, i])
        divider30 = make_axes_locatable(axs[4, i])
        cax10 = divider10.append_axes('right', size='5%', pad=0.3) # size is colorbar (CB) thickness; pad is CB-IMS separation
        cax20 = divider20.append_axes('right', size='8%', pad=0.3) 
        cax30 = divider30.append_axes('right', size='10%', pad=0.2) 
        #image1.set_clim(vmax=30) # vmin=0, COMMENT THIS LINE OUT TO SEE VARIETY WITHIN THE ALL-BLUE PLOTS
        cbar10= fig.colorbar(image10, cax=cax10, ticks=[np.min(res_rot_10mb[11, i,:,:]), 0, np.max(res_rot_10mb[11, i,:,:])])
        cbar20= fig.colorbar(image20, cax=cax20, ticks=[np.min(res_rot_20mb[11, i,:,:]), 0, np.max(res_rot_20mb[11, i,:,:])])
        cbar30= fig.colorbar(image30, cax=cax30, ticks=[np.min(res_rot_30mb[11, i,:,:]), 0, np.max(res_rot_30mb[11, i,:,:])])
        #cbar10.set_label('Cbar10')        #cbar20.set_label('Cbar20')     #cbar30.set_label('Cbar30')
        

    '''# res_rot plots over different Exposures... '''
    # res_rot plots over different Exposures...
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    fig.suptitle('Res_Rot over 4 different exposures (#2)')
    axs[0].imshow(res_rot[8, 4,:,:], origin='lower', cmap='Blues_r')
    axs[1].imshow(res_rot[8,12,:,:], origin='lower', cmap='Blues_r')
    axs[2].imshow(res_rot[8,20,:,:], origin='lower', cmap='Blues_r')
    axs[3].imshow(res_rot[8,28,:,:], origin='lower', cmap='Blues_r')
        
    
    # Save the figure
    print("Save rot Figure...")
    documents_path = "/home/dcpetit/Desktop/research_FPI/"
    filename_rot = "fpi_res_rot_figure.png" # or jpg
    full_path = documents_path + filename_rot #full_path = os.path.join(documents_path, filename)
    #plt.savefig(full_path) 
    #print(f"Figure saved to: {full_path}")
    
    plt.show()
    
    ############################################################################################################
    ### ### ### Above was res_raw/res_rot for 1 delay; below is res_raw for plotting 5 varied delays ### ### ###
    ############################################################################################################
    
    #MB = 40
    WLC_Plotted = 8 
    Xdelay = 4.46 # This is the optimal phase delay (obtained from the computation below) 
    Xdelay1, Xdelay2, Xdelay3 = 1.5, 2.5, 3.5 # (diameter/4)+1, (diameter/4)+2, (diameter/4)+3
    for i in range(WLC_num):
        wlDistribution.append(10*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, Xdelay, wlD, decay_factor) # Ydelay,
        CB_10, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB, AL, location, Xdelay1, wlD, DF)
        CB_20, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB, AL, location, Xdelay2, wlD, DF)
        CB_30, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB, AL, location, Xdelay3, wlD, DF)
        
        ### ### Align the CB, and then add it to the res_raw
        for j in range(len(angles_BPic20)):
            aligned_CB, rotMat = align_CB_to_angle(checkerboard, -1*angles_BPic20[j])                   # a small-ish 50,50 matrix
            aligned_CB_10, rotMat_10 = align_CB_to_angle(CB_10, -1*angles_BPic20[j])                    # a small-ish 50,50 matrix
            aligned_CB_20, rotMat_20 = align_CB_to_angle(CB_20, -1*angles_BPic20[j])                    # a small-ish 50,50 matrix
            aligned_CB_30, rotMat_30 = align_CB_to_angle(CB_30, -1*angles_BPic20[j])                    # a small-ish 50,50 matrix
            aligned_CB_neg, rotMat_neg = align_CB_to_angle(checkerboard, -1*angles_BPic20[-(j+1)])      # a small-ish 50,50 matrix
                    
            mitigated_signal = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB)                # a 200,200 image
            mitigated_signal_10 = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_10)          # a 200,200 image
            mitigated_signal_20 = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_20)          # a 200,200 image
            mitigated_signal_30 = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_30)          # a 200,200 image
            
            mitigated_signal_neg = add_wavy_checkerboard(res_raw[i,j,:,:], aligned_CB_neg)  # a 200,200 image
                
            # Assign values in the res_raw matrix
            res_raw[i,j,:,:] = mitigated_signal
            res_raw_10mb[i,j,:,:] = mitigated_signal_10
            res_raw_20mb[i,j,:,:] = mitigated_signal_20
            res_raw_30mb[i,j,:,:] = mitigated_signal_30
            res_raw_neg[i,j,:,:] = mitigated_signal_neg 
            
    fig, axs = plt.subplots(nrows=5, ncols=exposures, figsize=(exposures*6, exposures*0.8))
    fig.suptitle('Res Rot plots - WLC #'+str(WLC_Plotted)+', all maxBrights are = '+str(MB)+', varied phase delay') # or plt.suptitle('Res Raw plots')
    for i in range(exposures): # range(len(res_raw_original[0,:,0,0]))
        axs[0, i].title.set_text("Original - exp#" + str(i+1) + ", max=" + maxF(res_raw_original,WLC_Plotted,i,1) + ", min=" + minF(res_raw_original,WLC_Plotted,i,1))#+". min: "+str(np.round(np.min(mitigated_signal),2)) +". max: "+str(np.round(np.max(mitigated_signal), 2)) )
        axs[0, i].imshow(res_raw_original[WLC_Plotted, i,:,:], origin='lower', cmap='Blues_r')
        axs[1, i].title.set_text("Modified (Delay="+str(Xdelay)+") - exp#"+str(i+1) + ", max=" + maxF(res_raw,WLC_Plotted,i,1) + ", min=" + minF(res_raw,WLC_Plotted,i,1))
        axs[1, i].imshow(res_raw[WLC_Plotted, i,:,:], origin='lower', cmap='Blues_r')
        image10 = axs[2, i].imshow(res_raw_10mb[WLC_Plotted, i,:,:], origin='lower', cmap='Blues_r')
        axs[2, i].title.set_text("Delay= "+str(Xdelay1) + ", exp#" + str(i+1) + ", max=" + maxF(res_raw_10mb,WLC_Plotted,i,1) + ", min=" + minF(res_raw_10mb,WLC_Plotted,i,1))
        image20 = axs[3, i].imshow(res_raw_20mb[WLC_Plotted, i,:,:], origin='lower', cmap='Blues_r')
        axs[3, i].title.set_text("Delay= "+str(Xdelay2) + ", exp#" + str(i+1) + ", max=" + maxF(res_raw_20mb,WLC_Plotted,i,1) + ", min=" + minF(res_raw_20mb,WLC_Plotted,i,1))
        image30 = axs[4, i].imshow(res_raw_30mb[WLC_Plotted, i,:,:], origin='lower', cmap='Blues_r') #axs[2, i].imshow(res_raw_neg[11, i,:,:], origin='lower', cmap='Blues_r')
        axs[4, i].title.set_text("Delay= "+str(Xdelay3) + ", exp#" + str(i+1) + ", max=" + maxF(res_raw_30mb,WLC_Plotted,i,1) + ", min=" + minF(res_raw_30mb,WLC_Plotted,i,1))
        
        # Put color bars next to the above plots
        divider10 = make_axes_locatable(axs[2, i])
        divider20 = make_axes_locatable(axs[3, i])
        divider30 = make_axes_locatable(axs[4, i])
        cax10 = divider10.append_axes('right', size='5%', pad=0.3) # size is colorbar (CB) thickness; pad is CB-IMS separation
        cax20 = divider20.append_axes('right', size='8%', pad=0.3) 
        cax30 = divider30.append_axes('right', size='10%', pad=0.2) 
        #image1.set_clim(vmin=0, vmax=30) # COMMENT THIS LINE OUT TO SEE VARIETY WITHIN THE ALL-BLUE PLOTS
        clim_max10 = np.max(res_raw_10mb[WLC_Plotted, :,:,:]) # Change the i to :
        clim_max20 = np.max(res_raw_20mb[WLC_Plotted, :,:,:])
        clim_max30 = np.max(res_raw_30mb[WLC_Plotted, :,:,:])
        clim_max = np.max([clim_max10, clim_max20, clim_max30])
        '''
        image10.set_clim(vmin=0, vmax=clim_max)
        image20.set_clim(vmin=0, vmax=clim_max)
        image30.set_clim(vmin=0, vmax=clim_max)
        '''
        cbar10= fig.colorbar(image10, cax=cax10, ticks=[np.min(res_raw_10mb[WLC_Plotted, i,:,:]), 0, np.max(res_raw_10mb[WLC_Plotted, i,:,:])])
        cbar20= fig.colorbar(image20, cax=cax20, ticks=[np.min(res_raw_20mb[WLC_Plotted, i,:,:]), 0, np.max(res_raw_20mb[WLC_Plotted, i,:,:])])
        cbar30= fig.colorbar(image30, cax=cax30, ticks=[np.min(res_raw_30mb[WLC_Plotted, i,:,:]), 0, np.max(res_raw_30mb[WLC_Plotted, i,:,:])])
        #cbar10.set_label('Cbar10'), cbar20.set_label('Cbar20'), cbar30.set_label('Cbar30')
    '''
    fig.colorbar(image10, label=f"Values (max: {clim_max})") # ax=axes, 
    fig.colorbar(image20, label=f"Values (max: {clim_max})")
    fig.colorbar(image30, label=f"Values (max: {clim_max})")
    '''
    plt.show()
    print('The maxiumums of the 3 phase delays are:', clim_max10, clim_max20, clim_max30)
    print('The maxiumums of the 3 phase delays maximums is:', clim_max)

    # Save the figure
    print("Save delays Figure...")
    documents_path = "/home/dcpetit/Desktop/research_FPI/"
    filename_delay = "fpi_res_raw_figure_xdelays.png" # or jpg
    full_path = documents_path + filename_delay #full_path = os.path.join(documents_path, filename)
    #plt.savefig(full_path) 
    #print(f"Figure saved to: {full_path}")    
    
    ##########################################################################################################################
    ### ### ### Above was res_raw plotting 5 varied delays. Below is res_rot data analysis (many delays, no plots) ### ### ###
    ##########################################################################################################################

    print('starting a (long) computation... ... ...')
    delaysNum = 40 # When optimizing, make this big, maybe 1000 or so 
    xDelay = np.linspace(0, 2*np.pi, delaysNum)
    best_min, best_delay_idx, best_delay_rad = -1000, -1, -1

    ### Loop through the 39 WLCs
    for i in range(WLC_num):
        if i == np.rint(WLC_num/2):
            print('halfway there...')
        #What is this doing? Making the spacing between the entire 39 WLC range? 
        wlDistribution.append(10*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        #checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, Xdelay, wlD, decay_factor) # Ydelay, ---Do I need this? 7 March 2023

        ### Loop through the many phase delays
        for j in range(delaysNum): # xDelay 
            CB, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB, AL, location, xDelay[j], wlD, DF) # j

            ### Loop through the 32 exposures 
            for k in range(len(angles_BPic20)):
                
                #Align the CB (always at 0), and then add it to each res_rot exposure
                aligned_CB, rotMat = align_CB_to_angle(CB, 0)                                   # a small-ish 50,50 matrix                
                mitigated_signal = add_wavy_checkerboard(res_rot[i,k,:,:], aligned_CB)          # a 200,200 image
                
                # Assign values in the res_rot matrix (for this WLC and exposure), which now has a different phase delay... 
                #res_rot[i,k,:,:] = mitigated_signal
                
                # Compare values in it to all the earlier phase delayed results
                if np.abs(np.min(mitigated_signal)) < np.abs(best_min) and np.min(mitigated_signal) < 0:
                    best_min = np.round(np.min(mitigated_signal), 2)
                    best_delay_rad = xDelay[j]                   # ~6.3 radian
                    best_delay_idx = j                           # The last (or 2nd to last index...?)
                    best_WLC = i
                    best_exp = k
                    
    # Do data analysis on the many possible phase delays, find the one with the smallest minimum, compare to the original minimum
    # Before, used the 8th WLC and had 4.47 rad delay make a best minimum of -6.3
    print('We are going to use the variable: res_rot_original[best_WLC,best_exp,:,:]. The best_WLC,best_exp are:', best_WLC, best_exp)
    max_original = np.round(np.max(res_rot_original[best_WLC,best_exp,:,:]), 1) 
    min_original = np.round(np.min(res_rot_original[best_WLC,best_exp,:,:]), 1)
    best_delay_rad = np.round(best_delay_rad, 2)
    improvementPercent = np.round( (min_original - best_min) * 100 / min_original, 1)
    print('\nRegarding the best (WLC ='+str(best_WLC)+', exposure = '+str(best_exp)+') observation, with CB Max_B = '+str(MB)+'...\n')
    print('The original maximum and minimum are ' +str(max_original)+ ' and ' +str(min_original) + '.')
    print('As for the optimal phase...')
    print('At phase #' +str(best_delay_idx)+', ('+str(best_delay_rad)+' rad) there is the smallest minimum value of ' +str(best_min)+ '.')
    print('This indicates that ' +str(improvementPercent)+ ' % of the artifact (problem) has been solved\n') 
    print('Now plot the minimums for each xDelay given this WLC and exposure\nThe description of the plot should match the above\n')
    # At phase #711, (4.47 rad) there is the smallest minimum value of -6.3.    (other minimums=4.4,  rad)

    #################################################################################
    ### plot the relationship of minimum wrt phase delay (mitigated and original) ###
    #################################################################################
    wlD = wlDistribution[best_WLC]
    all_mins = []
    # Loop through the many delays (within 2pi)
    for j in range(delaysNum):
        CB, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB, AL, location, xDelay[j], wlD, DF)
        best_min = 1000
        # Loop through the 32 exposures
        for k in range(len(angles_BPic20)):
            # Align the CB (always at 0), and then add it to each res_rot exposure
            aligned_CB, rotMat = align_CB_to_angle(CB, 0)
            mitigated_signal = add_wavy_checkerboard(res_rot[best_WLC,k,:,:], aligned_CB)
            if np.abs(np.min(mitigated_signal)) < np.abs(best_min) and np.min(mitigated_signal) < 0: # If this has the best S-well in the ~100 delay
                best_min = np.min(mitigated_signal)
                best_delay_rad = xDelay[j]
                best_delay_idx = j
                best_exp = k

        # Add the minimum value of the image/matrix at this delay to a variable
        all_mins.append(best_min)
    best_min = str(np.round(best_min ,2))
    best_delay_idx = np.where(all_mins==np.max(all_mins))[0][0]
    print('best_delay_idx:', best_delay_idx)
    best_delay_rad = xDelay[best_delay_idx]
    if hasattr(best_delay_idx, "__len__"): #np.shape(best_delay_idx) > 1
        print('There are (probably) multiple maxima which yields arrays instead of constants :-(')
        print('np.shape(best_delay_idx)', np.shape(best_delay_idx))
        print('len(best_delay_idx)', len(best_delay_idx))
        best_delay_idx = best_delay_idx[0]
    #best_delay_rad = str(np.round(best_delay_rad,3))
    #best_delay_rad = all_mins[best_delay_idx]
    best_SigWell = np.max(all_mins)
    plt.plot(xDelay, all_mins, 'o', color='g', markersize=3)
    plt.vlines(xDelay[best_delay_idx], -5, -41, color='red', linestyle='dashed', linewidth=1)
    plt.xlabel('Phase Delay, applied to the CB [Radians]'), plt.ylabel('Minimum signal value (in the artifact-mitigated result)'), 
    plt.title('Phase Delay, WLC='+str(best_WLC)+' & exposure='+str(best_exp)+'. Best is $f(x)$ --> 0 ('+str(np.round(best_delay_rad,2))+' rad)')
    plt.show()

    print('the optimal phase...')
    print('best_SigWell:', np.round(best_SigWell,3))
    print('The length of all_mins should be ~100, and is: ', len(all_mins)) # Delete after March 2024
    print('At phase #' +str(best_delay_idx)+' and exposure='+str(best_exp)+', ('+str(np.round(best_delay_rad,2))+' rad) there is the smallest minimum value of ' +best_min+ '.')

    ###############################################################################################################################
    ### Make 1 plot, 2 lines: 1st is original data, 2nd is mitigated. x is exposure number, y is minimum pixel value. WLC=const ###
    ###############################################################################################################################
    ### redefine res_rot for the optimal phase delay (best_delay_rad)
    print('Compare the minimums of the exposures in original vs. mitigated data')
    best_difference = 0
    for i in range(WLC_num):
        wlDistribution.append(10*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        # Generate a checkerboard 
        checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, best_delay_rad, wlD, decay_factor)
        # Align the CB, and then add it to the res_rot
        for j in range(len(angles_BPic20)):
            aligned_CB, rotMat = align_CB_to_angle(checkerboard, 0)                               # a small-ish 50,50 matrix
            mitigated_signal = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB)                # a 200,200 image            
            res_rot[i,j,:,:] = mitigated_signal
            # this combo of i & j yields greatest increase in the mitigated minimum from the original minimum, save the WLC, then later plot it
            if np.min(res_rot[i,j,:,:]) - np.min(res_rot_original[i,j,:,:]) > best_difference:
                best_rr, best_rro = np.min(res_rot[i,j,:,:]), np.min(res_rot_original[i,j,:,:])
                best_difference = best_rr - best_rro
                best_diff_WLC = i
                best_diff_exp = j

    # define variables, add minimums to them  
    exposures_orig, exposures_miti = range(exposures), range(exposures)
    mins_orig, mins_miti = [], []
    for i in range(exposures):
        mins_orig.append(np.min(res_rot_original[best_WLC,i,:,:]))
        mins_miti.append(np.min(res_rot[best_WLC,i,:,:]))

    # Plot it
    plt.plot(exposures_orig, mins_orig, 'o', color='r', markersize=3.5, label='original')
    plt.plot(exposures_miti, mins_miti, 'o', color='g', markersize=2.5, label='mitigated')
    plt.xlabel('Exposure number'), plt.ylabel('Minimum signal value (in the exposure)')
    plt.title('Minimum values, for each exposure; WLC='+str(best_WLC)+'. with Delay= '+str(np.round(best_delay_rad,2))+' .') #Best is $f(x)$ --> 0 ('+str(np.round(best_delay_rad,2))+' rad)')
    plt.legend()
    plt.show()

    print('The best difference was going from '+str(best_rro)+' to '+str(best_rr)+' ('+str(best_difference)+')')
    print('and occured at WLC='+str(best_diff_WLC)+' and exposure #='+str(best_diff_exp)+'. Plotting this WLC/Exp...')
    mins_orig_opt, mins_miti_opt = [], []
    for i in range(exposures):
        mins_orig_opt.append(np.min(res_rot_original[best_diff_WLC,i,:,:]))
        mins_miti_opt.append(np.min(res_rot[best_diff_WLC,i,:,:]))
    plt.plot(exposures_orig, mins_orig, 'o', color='r', markersize=3.5, label='original')
    plt.plot(exposures_miti, mins_miti, 'o', color='g', markersize=2.5, label='mitigated')
    plt.xlabel('Exposure number'), plt.ylabel('Minimum signal value (in the exposure)')
    plt.title('Minimum values, for each exposure; WLC='+str(best_diff_WLC)+'. with (here an unoptimized) Delay= '+str(np.round(best_delay_rad,2))+' .') #Best is $f(x)$ --> 0 ('+str(np.round(best_delay_rad,2))+' rad)')
    plt.legend()
    plt.show()

    
    # There is little different between these original data minimums and the new mitigated data minimums...

    ############################################################################################################################################
    ### Calculate the average minimum, then for loop over phase delay and calculate average minimum for each delay, compare original w/ best ###
    ############################################################################################################################################
    avg_mins_orig = np.mean(mins_orig)

    delaysNum = 50
    xDelay = np.linspace(0, 2*np.pi, delaysNum)
    #best_min, best_delay_idx, best_delay_rad = -1000, -1, -1
    best_avg, best_delay_idx, best_delay_rad = -1000, -1, -1

    print('Begin calculating the best average minimum for many phase delays... ... ...') 
    # Loop through the 39 WLCs
    for i in range(WLC_num):
        #What is this doing? Making the spacing between the entire 39 WLC range? 
        wlDistribution.append(10*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        # Loop through the many phase delays
        for j in range(delaysNum): # xDelay 
            CB, CB4D = make_oscillating_checkerboard_noGap(D, x_squares, y_squares, center_square, MB, AL, location, xDelay[j], wlD, DF)
            exposure_mins = []
            # Loop through the 32 exposures 
            for k in range(len(angles_BPic20)):
                #Align the CB (always at 0), and add it to each res_rot exposure
                aligned_CB, rotMat = align_CB_to_angle(CB, 0)                
                mitigated_signal = add_wavy_checkerboard(res_rot[i,k,:,:], aligned_CB)
                # Add the minimum of this exposure to an array
                exposure_mins.append(np.min(mitigated_signal))
            # After each set of exposures compute average minimum of that delay, then, if it's better, save it as best
            avg_exposure_mins = np.mean(exposure_mins)
            if np.abs(avg_exposure_mins) < np.abs(best_avg) and avg_exposure_mins < 0:
                best_avg = avg_exposure_mins
                best_delay_rad = xDelay[j]
                best_delay_idx = j                   
                best_WLC = i
    print('\n\n\nFinished calculating best phase delay on average minimum value')
    print('The orginal 32 exposures had an average minimum of:\t\t', np.round(avg_mins_orig, 1))
    print('The best mitigated 32 exposures had an average minimum of:\t', np.round(best_avg, 1))
    print('Which occured at WLC='+str(best_WLC)+', Delay='+str(np.round(best_delay_rad,2))+' (rad?).')
    # Now plot this best delays' minimum with the original minimums. WLC=26, Delay=1.07.
    WLC = 26
    wlDistribution = (10*(wlMax - wavelength_channels[WLC])/(wlMax - wlMin) - 7.6) 
    wlDistrib_sin = (np.sin(wlDistribution))
    wlD = wlDistribution
    checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, best_delay_rad, wlD, decay_factor)
    for j in range(len(angles_BPic20)):
        aligned_CB, rotMat = align_CB_to_angle(checkerboard, 0)                               # a small-ish 50,50 matrix
        mitigated_signal = add_wavy_checkerboard(res_rot[WLC,j,:,:], aligned_CB)                # a 200,200 image            
        res_rot[WLC,j,:,:] = mitigated_signal
    mins_orig, mins_miti = [], []
    for i in range(exposures):
        mins_orig.append(np.min(res_rot_original[WLC,i,:,:]))
        mins_miti.append(np.min(res_rot[WLC,i,:,:]))

    # Plot it
    plt.plot(exposures_orig, mins_orig, 'o', color='r', markersize=3.5, label='original')
    plt.plot(exposures_miti, mins_miti, 'o', color='g', markersize=2.5, label='mitigated')
    plt.xlabel('Exposure number'), plt.ylabel('Minimum signal value (in the exposure)')
    plt.title('Minimum values, best phase delay, wrt exposure; WLC='+str(WLC)+', Delay='+str(np.round(best_delay_rad,2))+'.') #Best is $f(x)$ --> 0 ('+str(np.round(best_delay_rad,2))+' rad)')
    plt.legend()
    plt.show()
    # Results: The mitigated data has wells about 7.5 points shallower

    ############################################################################################################################################
    ### Calculate average minimum, then for loop over phase delay, calculate largest difference between original & mitigated average minimum ###
    ############################################################################################################################################



    #########################################################################################################################################
    ### Determine the location and intensity of the maximum pixel brigtness location (if it's B-D-Bright, ignore the locations where Dim) ###
    #########################################################################################################################################

    wlDistribution, wlDistrib_sin = [], []
    for i in range(WLC_num): # [8, 8]
        wlDistribution.append(10*(wlMax - wavelength_channels[i])/(wlMax - wlMin) - 7.6) 
        wlDistrib_sin.append(np.sin(wlDistribution[i]))
        wlD = wlDistribution[i]
        # Generate a checkerboard
        checkerboard, CB4D = make_oscillating_checkerboard_noGap(diameter, x_squares, y_squares, center_square, max_brightness, amp_nonRad_less, location, best_delay_rad, wlD, decay_factor)
        # Align the CB, and then add it to each of the res_rot 
        for j in range(len(angles_BPic20)):
            aligned_CB, rotMat = align_CB_to_angle(checkerboard, 0)                               # a small-ish 50,50 matrix
            mitigated_signal = add_wavy_checkerboard(res_rot[i,j,:,:], aligned_CB)                # a 200,200 image            
            res_rot[i,j,:,:] = mitigated_signal

    # define variables, add minimums to them
    exposures_orig, exposures_miti = range(exposures), range(exposures)
    max_bright_I, max_bright_loc = np.array([]), np.array([]) 
    for i in range(exposures):
        max_bright_I = np.append(max_bright_I, np.max(res_rot_original[best_WLC,i,:,:])) # using best_WLC from last section
        print('np.where(res_rot_original[best_WLC,i,:,:] == max_bright_I[i]) is:', np.where(res_rot_original[best_WLC,i,:,:] == max_bright_I[i]) )
        if len(np.where(res_rot_original[best_WLC,i,:,:] == max_bright_I[i])) > 2:
            print('Entered the if statement, it\'s (probably more than 1 optimal pixel???) length is > 2')
        max_bright_loc = np.append(max_bright_loc, np.where(res_rot_original[best_WLC,i,:,:] == max_bright_I[i])) # multiple ties for 1st place 

    # Plot it
    plt.plot(exposures_orig, max_bright_I, 'o', color='g', markersize=3.5, label='WLC=8')
    #plt.plot(exposures_miti, mins_miti, 'o', color='g', markersize=2.5, label='mitigated')
    plt.xlabel('Exposure number'), plt.ylabel('Maximum signal value (in the exposure)')
    plt.title('Maximum Brightenss - Intensity; WLC='+str(best_WLC)+'. with Delay= '+str(np.round(best_delay_rad, 2))+'.') # Best is $f(x)$ --> 0 ('+str(np.round(best_delay_rad,2))+' rad)')
    plt.legend()
    plt.show()


    # Colormap defining a sequence of colors
    cmap = plt.colormaps['viridis']
    #cmap = plt.cm.get_cmap('viridis')
    
    # Normalize z for color mapping
    e_norm = (np.subtract(exposures_orig, np.min(exposures_orig)) ) / (np.max(exposures_orig) - np.min(exposures_orig) ) # delete???
    #print('np.shape(max_bright_loc[0,1]))', np.shape(max_bright_loc[0,1]))
    colors = cmap(np.linspace(0, 1, num=32)) #len(max_bright_loc[0,1])))
    print('e_norm is:', e_norm) 

    max_bright_loc = max_bright_loc.reshape(32, 2) 
    '''
    print('np.shape(max_bright_loc) is:', np.shape(max_bright_loc))
    print('max_bright_loc is:', max_bright_loc)
    print('max_bright_loc[:,1] is:', max_bright_loc[:,1])
    '''

    # Create scatter plot with color based on z
    for i in range(len(max_bright_loc[:,0])):
        size = -(6/32)*i + (6 + 1) # where 6(+1) is the maximum size, 32 is the number of exposures, and 1 is the minimum size 
        plot_max = plt.plot(max_bright_loc[i,0], max_bright_loc[i,1], 'o',  c=colors[i], markersize=size, label=str(i+1)) #  cmap=cmap, #scatter = 
    #plt.colorbar(plot_max, label='exposure #')
    plt.ylim([np.min(max_bright_loc[:,1])-6, np.max(max_bright_loc[:,1])+6])
    plt.xlabel('X') 
    plt.ylabel('Y')
    plt.title('Locations of Maximum Signal (over 32 exposures in color gradient)') #; WLC=8, not:'+str(best_WLC)+'. with Delay= '+str(best_delay_rad)+' .') # Best is $f(x)$ --> 0 ('+str(np.round(best_delay_rad,2))+' rad)')
    plt.legend(ncol=2, bbox_to_anchor=(1.4, 1)) # Shows the exposure number (by color gradient) # (1.3=x_dist, 1.1=y_dist)
    plt.show()
    #plt.plot(exposures_miti, mins_miti, 'o', color='g', markersize=2.5, label='mitigated')

    # Give the next

    '''
    # For testing purposes, revert the res_raw and res_rot to their original values (use when changing magnitudes or locations of FPI)
    res_raw = res_raw_original
    res_rot = res_rot_original 
    '''
    return res_raw, res_rot
