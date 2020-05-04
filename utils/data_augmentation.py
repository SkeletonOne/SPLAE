import numpy as np
import SimpleITK as sitk

def smooth_image(img, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    img = sitk.CurvatureFlow(image1=img,
                                    timeStep=t_step,
                                    numberOfIterations=n_iter)

    return img

def Normailize_images(images):
    images = np.concatenate(images , axis=0).reshape(-1, 512, 512)
    mu = np.mean(images)
    sigma = np.std(images)
    images = (images - mu)/sigma
    return images