# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
'''
	nms.py: CPU implementation of non maximal supression modified from Ross's code.
	Authors : svp
	Modified from https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
	to accommodate a corner case which handles one box lying completely inside another.
'''
import numpy as np


def is_square(inter, areas):
    truth_val = np.logical_not((np.logical_and((np.sqrt(areas) ** 2 == areas), (np.sqrt(inter) ** 2 == inter))))
    return np.float32(truth_val)


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        remove_index_1 = np.where(areas[i] == inter)  # i is included by others
        remove_index_2 = np.where(areas[order[1:]] == inter)  # i include pthers

        ovr = 1 / 3 * inter / (areas[i] + areas[order[1:]] - inter) \
              + 1 / 3 * inter / areas[i] \
              + 1 / 3 * inter / areas[order[1:]]

        # ovr = inter / (areas[i] + areas[order[1:]] - inter)* np.maximum (areas[order[1:]]/areas[i], areas[i]/areas[order[1:]])

        ovr[remove_index_1] = 1.0
        ovr[remove_index_2] = 1.0
        inds = np.where(ovr <= thresh)[0]  # get the index(a series)
        order = order[inds + 1]

    return keep


if __name__ == '__main__':
    a = np.array([[1, 2, 4, 5, 0.9], [1, 2, 3, 4, 0.99], [8, 2, 9, 4, 0.99]])
    keep = nms(a, 0.2)
    print(keep)
    np.where(np.array([78, 3, 4, 54, 3, ]) > 10)