import pdb
import cv2
import numpy as np
import os
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
imgid = '1202'
resolution=118
name = 'finalx32'  # 'stem' 'layer1' 'finalx4' 'finalx8' , 'finalx16', 'finalx32'



os.makedirs('./exp/moe/{}'.format(imgid),exist_ok=True)

def imgwithcam(img, cam_data, imgid, resolution, cls ):
    result = overlay_mask(to_pil_image(img), to_pil_image(cam_data, mode='F'), alpha=0.5)
    result = cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2BGR)
    cv2.imwrite('./exp/moe/{}/{}_{}_cam.png'.format(imgid, resolution,cls), result)


def vis4feature_sum():
    # pdb.set_trace()
    data = np.load('./exp/moe/current_{}.npy'.format(resolution))


    print(data, data.shape)
    data_sum = data.mean(1).squeeze(0)
    data_sum=data_sum[50:,:]
    pred_color_map = cv2.applyColorMap(
        (255 * data_sum / (data_sum.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    cv2.imwrite('./exp/moe/vis_{}_{}_{}.png'.format(imgid, resolution, 'sum'), pred_color_map)

def vis4feature_single_channel():
    data = np.load('./exp/moe/{}_{}.npy'.format(imgid, resolution))
    _, c, h, w = data.shape
    img = read_image("../ProcessedData/QNRF/images/{}.jpg".format(imgid))

    for i in range(c):
        data_c = data[:, i]
        cam_data = data_c.squeeze(0)
        pred_color_map = cv2.applyColorMap(
            (255 * data_c / (data_c.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        cv2.imwrite('./exp/moe/{}/{}_{}.png'.format(imgid, resolution,i), pred_color_map)
        imgwithcam(img,cam_data, imgid, resolution, i)


def moe_label_to_mask():
    import torch
    resolution = 2
    data = np.load('./exp/moe/{}.npy'.format('29'))
    data_1 = np.zeros(data.shape).astype(int)
    for i in range(resolution):
        # data_1 += (data==i).astype(int)
        data_1 =  (data==3).astype(int) +(data==2).astype(int)+(data==1).astype(int)
    data_1 = F.interpolate(torch.from_numpy(data_1).float(), scale_factor=128)
    data_1 = 1- data_1.numpy().squeeze()
    # data_1 =1-data_1
    B = (data_1*184)[:,:,None]
    G = (data_1*193)[:,:,None]
    R = (data_1*250)[:,:,None]
    BGR = np.concatenate([B,G,R],2).astype(np.uint8)

    # print(data)
    #
    # pred_color_map = cv2.applyColorMap(
    # (255 * data_c / (data_c.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    #

    cv2.imwrite('./exp/moe/{}/{}_{}.png'.format(imgid,'moe_lable_mask', resolution), BGR)
    pdb.set_trace()
    # imgwithcam(img,cam_data, imgid, resolution, i)
moe_label_to_mask()
# vis4feature_sum()
# vis4feature_single_channel()
    # pdb.set_trace()