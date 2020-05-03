import os
import numpy as np
import cv2
import SimpleITK as sitk


def generate_2D_imgs(file_path, save_path):
    cases = []
    masks = []
    count = 0
    for dirName, subdirList, fileList in os.walk(file_path):
        for filename in fileList:
            if ".mhd" in filename.lower():
                if len(filename) > 10:
                    masks.append(filename)
                else:
                    cases.append(filename)
            masks = sorted(masks)
            cases = sorted(cases)
        for case_num in range(len(cases)):
            case = sitk.ReadImage(dirName+'/'+cases[case_num])
            case = sitk.GetArrayFromImage(case)
            case_mask = sitk.ReadImage(dirName+'/'+masks[case_num])
            case_mask = sitk.GetArrayFromImage(case_mask)
            assert case.shape == case_mask.shape, 'IMG and GT do not have the same shape!'
            for img_num in range(case.shape[0]):
                img = case[img_num]
                gt = case_mask[img_num]
                img = np.array(cv2.resize(img,(512,512), interpolation=cv2.INTER_AREA),dtype='int16')
                cv2.imwrite(save_path+str(count)+'.png',img)
                gt = np.array(cv2.resize(gt,(512,512), interpolation=cv2.INTER_NEAREST),'int8')
                gt *= 255
                gt = gt.astype(np.uint8)
                cv2.imwrite(save_path+str(count)+'_mask.png',gt)
                count += 1
    return None