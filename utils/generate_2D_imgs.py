import os
import numpy as np
import cv2
import SimpleITK as sitk


def generate_2D_imgs(file_path, save_path):
    cases = []
    masks = []
    count = 0
    cases_val = ['Case05.mhd','Case15.mhd','Case25.mhd','Case35.mhd','Case45.mhd']
    masks_val = ['Case05_segmentation.mhd','Case15_segmentation.mhd','Case25_segmentation.mhd','Case35_segmentation.mhd','Case45_segmentation.mhd']
    for dirName, subdirList, fileList in os.walk(file_path):
        for filename in fileList:
            if ".mhd" in filename.lower():
                if len(filename) > 10:
                    masks.append(filename)
                else:
                    cases.append(filename)
    masks = sorted(masks)
    cases = sorted(cases)
    cases_train = sorted(list(set(cases).difference(set(cases_val))))
    masks_train = sorted(list(set(masks).difference(set(masks_val))))
    for case_num in range(len(cases_train)):
        case = sitk.ReadImage(dirName+'/'+cases_train[case_num])
        case = sitk.GetArrayFromImage(case)
        case_mask = sitk.ReadImage(dirName+'/'+masks_train[case_num])
        case_mask = sitk.GetArrayFromImage(case_mask)
        assert case.shape == case_mask.shape, 'IMG and GT do not have the same shape!'
        for img_num in range(case.shape[0]):
            img = case[img_num]
            gt = case_mask[img_num]
            img = np.array(cv2.resize(img,(320,320), interpolation=cv2.INTER_AREA),dtype='int16')
            cv2.imwrite('./imgs/'+str(count)+'.png',img)
            gt = np.array(cv2.resize(gt,(320,320), interpolation=cv2.INTER_NEAREST),'int8')
            gt *= 255
            gt = gt.astype(np.uint8)
            cv2.imwrite('./imgs/'+str(count)+'_mask.png',gt)
            count += 1
    print('train nums',count)
    for case_num in range(len(cases_val)):
        case = sitk.ReadImage(dirName+'/'+cases_val[case_num])
        case = sitk.GetArrayFromImage(case)
        case_mask = sitk.ReadImage(dirName+'/'+masks_val[case_num])
        case_mask = sitk.GetArrayFromImage(case_mask)
        assert case.shape == case_mask.shape, 'IMG and GT do not have the same shape!'
        for img_num in range(case.shape[0]):
            img = case[img_num]
            gt = case_mask[img_num]
            img = np.array(cv2.resize(img,(320,320), interpolation=cv2.INTER_AREA),dtype='int16')
            cv2.imwrite('./imgs/'+str(count)+'.png',img)
            gt = np.array(cv2.resize(gt,(320,320), interpolation=cv2.INTER_NEAREST),'int8')
            gt *= 255
            gt = gt.astype(np.uint8)
            cv2.imwrite('./imgs/'+str(count)+'_mask.png',gt)
            count += 1
    print('total nums',count)

    return None