import numpy as np
import SimpleITK as sitk
import torch
from skimage import exposure

# Define some preprocessing metrics when generating data, called in generate_2D_imgs.py

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    I am not familiar with sitk.CurvatureFlow. What is t_step?
    """
    print('Smoothing the images. This may take a while.')
    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)


    return imgs

def hist_equal(imgs):
    '''
    Input:
        imgs: a numpy array with (N, w, h)
    Output:
        imgs: a numpy array with (N, w, h), which is the hist-equalized input image.
    '''
    for mm in range(len(imgs)):
        imgs[mm] = exposure.equalize_hist(imgs[mm])*256
    return imgs