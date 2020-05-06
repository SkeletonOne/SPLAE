import os
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset

class TumorDataset(Dataset):
    '''
    Returns a TumorDataset class object which represents our tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    '''

    def __init__(self, root_dir, DEBUG = False):
        '''
        Constructor for our TumorDataset class.
        Parameters:
            root_dir(str): Directory with all the images.
            DEBUG(bool): To switch to debug mode for image transformation.

        Returns: None
        '''
        self.root_dir = root_dir
        # The default transformation is composed of 
        # 1) a grayscale conversion and 2) a resizing to input_length x input_length.
        self.default_transformation = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.Resize((input_length, input_length))
        ])
        self.DEBUG = DEBUG

    def __getitem__(self, index):
        '''
        Overridden method from inheritted class to support
        indexing of dataset such that datset[I] can be used
        to get Ith sample.
        Parameters:
            index(int): Index of the dataset sample

        Return:
            sample(dict): Contains the index, image, mask torch.Tensor.
                        'index': Index of the image.
                        'image': Contains the tumor image torch.Tensor.
                        'mask' : Contains the mask image torch.Tensor.
        '''
        # Find the filenames for the tumor images and masks.
        image_name = os.path.join(self.root_dir, str(index) + '.png')
        mask_name = os.path.join(self.root_dir, str(index) + '_mask.png')

        # Use PIL to open the images and masks.
        image = Image.open(image_name)
        mask = Image.open(mask_name)

        # Apply the default transformations on the images and masks.
        image = self.default_transformation(image)
        mask = self.default_transformation(mask)

        # Convert the images and masks to tensor.
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Construct the images and masks together in the form of a dictionary.
        sample = {'index': int(index), 'image': image, 'mask': mask}
        return sample

    def __len__(self):
        ''' Overridden method from inheritted class so that
        len(self) returns the size of the dataset.
        '''
        error_msg = 'Part of dataset is missing!\nNumber of tumor and mask images are not same.'
        total_files = len(glob(os.path.join(self.root_dir, '*.png')))

        # Sanity check: the number of files shall be even since tumor images and masks are in pairs.
        assert total_files % 2 == 0, error_msg + ' total files: %s' % (total_files)
        
        # Return how many image-mask pairs we have.
        return total_files // 2

# Total 1377 imgs. 1250+1250(rot) Train, 127 val/test. The 127 belongs to Case 05,15,25,35,45.
def get_indices(ltrain, lval, ltest):
    '''
    Gets the Training & Testing data indices for the dataset.
    Stores the indices and returns them back when the same dataset is used.
    Inputs:
        length(int): Length of the dataset used.
        val_split: the portion (0 to 1) of data used for validation.
        test_split: the portion (0 to 1) of data used for testing.
    Parameters:
        train_indices(list): Array of indices used for training purpose.
        validation_indices(list): Array of indices used for validation purpose.
        test_indices(list): Array of indices used for testing purpose.
    '''
    data = dict()
    indices = list(range(ltrain+lval+ltest))
    # currently using val=test, maybe will change it further.
    train_indices, validation_indices, test_indices = indices[:ltrain], indices[ltrain:ltrain+lval], indices[ltrain:ltrain+lval]
    np.random.shuffle(train_indices)
    np.random.shuffle(validation_indices)
    np.random.shuffle(test_indices)
    return train_indices, validation_indices, test_indices