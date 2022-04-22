from calendar import EPOCH
from operator import gt, mod
import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import os
import math
import skimage
import skimage.measure
import skimage.metrics


def cal_ssim(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2,multichannel = True,gaussian_weights=True, sigma = 1.5)

def cal_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def main(output_path,gt_path,dataset):
    outs = os.listdir(output_path)
    # BP_phantom_16_119_fake_B.png
    # BP_phantom_GT_106.png
    ssim, psnr = 0, 0
    outs = [out for out in outs if dataset in out]
    for out in outs:
        index = out.split('_')[3]
        if int(index) > 274:
            continue
        out_img = cv2.imread(os.path.join(output_path, out),0)
        gt_img = cv2.imread(os.path.join(gt_path, '%s_gt_%s.png'%(dataset, index)),0)  #BP_phantom_GT_%s.png
        gt_img = cv2.resize(gt_img, (256,256))

        ssim += cal_ssim(out_img, gt_img)
        psnr += cal_psnr(out_img, gt_img)
    ssim, psnr = ssim / len(outs), psnr / len(outs)
    print(output_path)
    print(ssim, psnr)


if __name__ == '__main__':
    model_name = '0414-10'
    epoch = '150'
    dataset = 'v_phantom' #'v_phantom'  #'BP_phantom'
    gt_path = 'datasets/mice_sparse8/testB'
    output_path = 'results/%s/test_%s/images'%(model_name, epoch)
    print(model_name, epoch, dataset)
    main(output_path,gt_path,dataset)