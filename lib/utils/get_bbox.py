import numpy as np
import torch
import torch.nn.functional as F
from .nms import *
import torch.nn as nn
import cv2
import pdb
def local_maximum(sub_pre,sub_bin, scale_factor=1.):
    sub_pre = torch.from_numpy(sub_pre).unsqueeze(0).unsqueeze(0)
    max_value = torch.max(sub_pre)

    # kernel = [[1/9., 1/9., 1/9.], [1/9., 1/9., 1/9.], [1/9., 1/9.,1/9.]]
    # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    # weight = nn.Parameter(data=kernel, requires_grad=False)
    # sub_pre = F.conv2d(sub_pre, weight, stride=1,padding=1)

    keep = nn.functional.max_pool2d(sub_pre, (3, 3), stride=1, padding=1)
    keep = (keep == sub_pre).float()
    sub_pre = keep * sub_pre

    sub_pre[sub_pre < 0.5 * max_value] = 0
    sub_pre[sub_pre > 0] = 1
    count = int(torch.sum(sub_pre).item())

    kpoint = sub_pre.data.squeeze(0).squeeze(0).cpu().numpy()

    points = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0]))).astype(np.float32)
    # distance_map = cv2.distanceTransform(sub_bin, cv2.DIST_L1,3)

    boxes = np.zeros((len(points), 5)).astype(np.float32)
    for i in range(len(points)):
        x, y = points[i]
        length =  scale_factor # max(distance_map[int(y), int(x)], scale_factor)
        boxes[i] = [x - length, y - length, 2*length, 2*length, 4*length*length]
    pre_data = {'num': count, 'points': points, 'boxes': boxes}
    return pre_data

def Noise_box_detection(recs):
    maintain_list = []
    recs[:, 2] = recs[:, 0] + recs[:, 2]
    recs[:, 3] = recs[:, 1] + recs[:, 3]
    length = len(recs)

    for i in range(length):
        if i < length - 1:
            j = i + 1
            index = (recs[i][0] >= recs[j:][:, 0]) & (recs[i][1] >= recs[j:][:, 1]) \
                    & (recs[i][2] <= recs[j:][:, 2]) & (recs[i][3] <= recs[j:][:, 3])
            index = np.where(index == True)[0]
            if index.size > 0:
                continue
            else:
                maintain_list.append(i)
        else:
            maintain_list.append(i)
    return maintain_list

def get_boxInfo_from_Binar_map(pred_map , threshold = 0.3,  min_area=4,scale_factor = 1., polish =False):
    # import pdb
    # pdb.set_trace()
    a = torch.ones_like(pred_map)
    b = torch.zeros_like(pred_map)
    Binar_map = torch.where(pred_map >= threshold, a, b).cpu().numpy()

    Binar_map = Binar_map.squeeze().astype(np.uint8)
    pred_map = pred_map.squeeze()
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_map, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :].astype(np.float32)
    points = centroids[1:, :].astype(np.float32)
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]

    order = np.argsort(boxes[:, 4])
    points = points[order]
    boxes = boxes[order]

    maintain_list = Noise_box_detection(boxes.copy())
    boxes = boxes[maintain_list]
    points = points[maintain_list]

    if polish:
        boxes_app = []
        points_app = []
        for id in range(len(boxes)):
            w_s, h_s, w, h, area = boxes[id]
            sub_pre = pred_map[int(h_s):int(h_s) + int(h), int(w_s):int(w_s) + int(w)].copy()
            sub_bin = Binar_map[int(h_s):int(h_s) + int(h), int(w_s):int(w_s) + int(w)].copy()
            iou = boxes[id, 4] / (w * h)
            ration = h / w
            if  area>20:
                if ration > 2 or ration < 0.5 or iou < 0.75:
                    pred_data = local_maximum(sub_pre,sub_bin,scale_factor)
                    if pred_data['num'] >= 1:
                        pred_data['boxes'][:, 0] += w_s
                        pred_data['boxes'][:, 1] += h_s
                        pred_data['points'][:, 0] += int(w_s)
                        pred_data['points'][:, 1] += int(h_s)
                        boxes[id, :] = pred_data['boxes'][0, :]
                        points[id, :] = pred_data['points'][0, :]

                        for k in range(1, pred_data['num']):
                            boxes_app.append(pred_data['boxes'][k, :])
                            points_app.append(pred_data['points'][k, :])

        # print('original:{}, add_boxes:{}, final_boxes:{}'.format(len(boxes), len(boxes_app), len(boxes) + len(boxes_app)))

        if len(boxes_app) > 0:
            boxes = np.concatenate((boxes, np.array(boxes_app)))
            points = np.concatenate((points, np.array(points_app).astype(np.int32)))
    new_boxes = np.zeros((len(points), 4)).astype(np.float32)
    scores = np.zeros((len(points), 1)).astype(np.float32)

    # for i in range(len(boxes)):
    #     x_s, y_s, w, h, area = boxes[i]
    #     x, y = points[i]
    #     # _scale = scale_map[y_s:y_s + h, x_s:x_s + w]
    #     # _mask = Binar_map[y_s:y_s + h, x_s:x_s + w]
    #     _pred = pred_map[int(y_s):int(y_s) + int(h), int(x_s):int(x_s) + int(w)]
    #     score =pred_map[int(y),int(x)]
    #     sigma = np.sqrt(w ** 2 + h ** 2)
    #     sin = h / sigma
    #     cos = w / sigma
    #
    #     scale = max( scale_map[int(y),int(x)], sigma / 2)  #if  index.sum()>0 else sigma / 2 #_scale[index].max()
    #
    #     de_h,  de_w = scale * sin, scale * cos
    #     new_x_s, new_x_e = x - de_w, x + de_w
    #     new_y_s, new_y_e = y - de_h, y + de_h
    #     new_boxes[i] = [new_x_s, new_y_s, new_x_e, new_y_e]
    #     scores[i] = score

    batch_id = np.zeros((len(points), 1)).astype(np.float32)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    boxes = boxes[:, :4]
    boxes = np.hstack((batch_id,boxes / scale_factor))
    boxes = torch.from_numpy(boxes)
    # boxes = np.hstack((boxes/scale_factor,scores))
    # new_boxes = np.hstack((new_boxes/scale_factor, scores))

    if polish:
        keep = nms(new_boxes,thresh=0.3)
        points = points[keep]
        boxes = boxes[keep]
        new_boxes = new_boxes[keep]
    pred_data = {'num': len(points), 'points': points/scale_factor, 'rois': boxes, 'new_boxes': new_boxes}
    return pred_data

def multiscale_nms(pred_data):

    base_boxes = pred_data[1]['boxes']
    base_points = pred_data[1]['points']
    base_new_boxes = pred_data[1]['new_boxes']

    for scale in pred_data.keys():
        if scale == 1:
            continue
        boxes = pred_data[scale]['boxes']
        points = pred_data[scale]['points']
        new_boxes = pred_data[scale]['new_boxes']

        base_boxes=  np.concatenate((base_boxes, boxes))
        base_points= np.concatenate((base_points, points))
        base_new_boxes = np.concatenate((base_new_boxes, new_boxes))

    # order = np.argsort((base_new_boxes[:, 3]-base_new_boxes[:, 1])*(base_new_boxes[:, 2]-base_new_boxes[:, 0]))
    # base_points = base_points[order]
    # base_boxes = base_boxes[order]
    # base_new_boxes = base_new_boxes[order]
    # #
    # keep = Noise_box_detection(base_new_boxes.copy())
    # base_points = base_points[keep]
    # base_boxes = base_boxes[keep]
    # base_new_boxes = base_new_boxes[keep]

    keep = nms(base_new_boxes,thresh=0.2)
    base_points = base_points[keep]
    base_boxes = base_boxes[keep]
    base_new_boxes = base_new_boxes[keep]

    pred_data = {'num': len(base_points), 'points': base_points , 'rois': base_boxes, 'new_boxes': base_new_boxes}
    return pred_data