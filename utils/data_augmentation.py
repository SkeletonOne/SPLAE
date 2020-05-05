import numpy as np
import SimpleITK as sitk
import torch
from skimage import exposure

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
        img = sitk.GetImageFromArray(imgs[mm])
        img = exposure.equalize_hist(img)
        imgs[mm] = sitk.GetArrayFromImage(img)
    return imgs

def randomflipping(imgs):
    pass

def rotation(imgs):
    pass