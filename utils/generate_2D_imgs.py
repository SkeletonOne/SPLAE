import os
import numpy as np
import cv2
import SimpleITK as sitk
import skimage.io as io

def loadImages(filename, plugin='simpleitk'):
    imagesArray=[];
    images=io.imread(filename,plugin=plugin)   
    #Resizing and stacking the slices
    for i in range(images.shape[0]):
        imagesArray.append(np.array(cv2.resize(images[i],(512,512), interpolation=cv2.INTER_AREA),dtype='int16'))
    return imagesArray

def generate_2D_imgs(file_path, save_path, do_normalize = True):
    cases = []
    masks = []
    count = 0
    xTrain=[]
    xVal=[]
    
    # Make four lists: cases_train, masks_train, cases_val, masks_val. Containing the .mhd files.
    for dirName, subdirList, fileList in os.walk(file_path):
        for filename in fileList:
            if ".mhd" in filename.lower():
                if len(filename) > 10:
                    masks.append(filename)
                else:
                    cases.append(filename)
    masks = sorted(masks)
    cases = sorted(cases)
    cases_val = ['Case05.mhd','Case15.mhd','Case25.mhd','Case35.mhd','Case45.mhd']
    masks_val = ['Case05_segmentation.mhd','Case15_segmentation.mhd','Case25_segmentation.mhd','Case35_segmentation.mhd','Case45_segmentation.mhd']
    cases_train = sorted(list(set(cases).difference(set(cases_val))))
    masks_train = sorted(list(set(masks).difference(set(masks_val))))

    # Make train imgs and gts
    for case_num in range(len(cases_train)):
        xTrain.extend(loadImages(dirName+'/'+cases_train[case_num]))
        case_mask = sitk.ReadImage(dirName+'/'+masks_train[case_num])
        case_mask = sitk.GetArrayFromImage(case_mask)
        for img_num in range(case_mask.shape[0]):
            gt = case_mask[img_num]
            gt = np.array(cv2.resize(gt,(512,512), interpolation=cv2.INTER_NEAREST),'int8')
            gt *= 255
            gt = gt.astype(np.uint8)
            cv2.imwrite('./imgs/'+str(count)+'_mask.png',gt)
            count += 1
    if do_normalize:
        xTrainMean=np.array(xTrain).mean()
        xTrainStd=np.array(xTrain).std()
        xTrain=(np.array(xTrain)-xTrainMean)/xTrainStd
    # Save train imgs
    for img_num, img in enumerate(xTrain):
        cv2.imwrite('./imgs/'+str(img_num)+'.png',img)
    print('train nums',count)
    
    # Make val imgs and gts
    for case_num in range(len(cases_val)):
        xVal.extend(loadImages(dirName+'/'+cases_val[case_num]))
        case_mask = sitk.ReadImage(dirName+'/'+masks_val[case_num])
        case_mask = sitk.GetArrayFromImage(case_mask)
        for img_num in range(case_mask.shape[0]):
            gt = case_mask[img_num]
            gt = np.array(cv2.resize(gt,(512,512), interpolation=cv2.INTER_NEAREST),'int8')
            gt *= 255
            gt = gt.astype(np.uint8)
            cv2.imwrite('./imgs/'+str(count)+'_mask.png',gt)
            count += 1
    if do_normalize:
        xVal=(np.array(xVal)-xTrainMean)/xTrainStd
    # Save val imgs
    for img_num, img in enumerate(xVal):
        cv2.imwrite('./imgs/'+str(img_num + 1250)+'.png',img)
    print('total nums',count)

    return None