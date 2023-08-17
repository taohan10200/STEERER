from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from .utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter
from scipy import spatial as ss
import json
import  os
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
def eval_boxes_AP(pred_boxes, gt_boxes, iou_thr=0.5, level=None):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
                - "bboxes": numpy array of shape (n, 4)
                - "labels": numpy array of shape (n, )
                - "bboxes_ignore" (optional): numpy array of shape (k, 4)
                - "labels_ignore" (optional): numpy array of shape (k, )
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
        """Calculate the ious between each bbox of bboxes1 and bboxes2.

        Args:
            bboxes1(ndarray): shape (n, 4)
            bboxes2(ndarray): shape (k, 4)
            mode(str): iou (intersection over union) or iof (intersection
                over foreground)

        Returns:
            ious(ndarray): shape (n, k)
        """

        assert mode in ['iou', 'iof']

        bboxes1 = bboxes1.astype(np.float32)
        bboxes2 = bboxes2.astype(np.float32)
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        ious = np.zeros((rows, cols), dtype=np.float32)
        if rows * cols == 0:
            return ious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            ious = np.zeros((cols, rows), dtype=np.float32)
            exchange = True
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
        for i in range(bboxes1.shape[0]):
            x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
            y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
            x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
            y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
            overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
                y_end - y_start + 1, 0)
            if mode == 'iou':
                union = area1[i] + area2 - overlap
            else:
                union = area1[i] if not exchange else area2
            ious[i, :] = overlap / (union+ 1e-20)
        if exchange:
            ious = ious.T
        return ious

    def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, iou_thr, level):

        for i_pred_p in range(pred_num):
            pred_dist = dist_matrix[i_pred_p, :]
            match_matrix[i_pred_p, :] = pred_dist >= iou_thr

        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
        tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
        tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

        level_list = level[tp_gt_index]

        # tp = tp_pred_index.shape[0]
        # fp = fp_pred_index.shape[0]
        # fn = fn_gt_index.shape[0]
        tp_c = np.zeros([num_classes])
        fn_c = np.zeros([num_classes])

        for i_class in range(num_classes):
            tp_c[i_class] = (level[tp_gt_index]==i_class).sum()
            fn_c[i_class] = (level[fn_gt_index]==i_class).sum()

        # assert  tp_gt_index.max() < dist_matrix.shape[0]
        return tp_pred_index , tp_c, fn_c  # tp, fp, fn,

    num_classes = 6
    mask_and_score = np.zeros((len(pred_boxes), 2), dtype=pred_boxes.dtype)
    fn_c = np.zeros([num_classes])
    tp_c = np.zeros([num_classes])

    if len(pred_boxes) == 0 and len(gt_boxes) != 0:
        fn_gt_index = np.array(range(gt_boxes.shape[0]))
        for i_class in range(num_classes):
            fn_c[i_class] = (level[fn_gt_index] == i_class).sum()

    if len(gt_boxes) != 0 and len(pred_boxes) != 0:
        mask_and_score[:, 1] = pred_boxes[:, 4]
        pred_boxes = pred_boxes[:, :4]
        ious_matrix = bbox_overlaps(pred_boxes, gt_boxes)
        assert  ious_matrix.shape[0] == pred_boxes.shape[0]
        match_matrix = np.zeros(ious_matrix.shape, dtype=bool)
        tp_fp_pred_index, tp_c, fn_c = compute_metrics(ious_matrix, match_matrix, pred_boxes.shape[0], gt_boxes.shape[0], iou_thr, level)

        # assert  tp_fp_pred_index.max() <= pred_boxes.shape[0]

        mask_and_score[tp_fp_pred_index, 0] = 1
    return mask_and_score, tp_c, fn_c

def compute_AP( mask_and_score_list, num_gt, thresh_num=1000):
    import pdb
    mask_score_dataset = np.vstack(mask_and_score_list)
    AP = 0
    pr_cruve = np.zeros((thresh_num, 2), dtype=np.float)
    length = mask_score_dataset.shape[0]
    if mask_score_dataset.shape[0]==0:
        return  AP, pr_cruve
    else:
        min_score, max_score = min(mask_score_dataset[:, 1]), max(mask_score_dataset[:, 1])
        mask_score_dataset[:, 1] = (mask_score_dataset[:, 1] - min_score) / (max_score - min_score)

        for i in range(thresh_num):
            idx = np.array(np.where(mask_score_dataset[:, 1] >= 1 - ((i + 1.) / thresh_num)))[0]
            tp = mask_score_dataset[idx, 0].sum()
            pr_cruve[i] = [tp / (idx.shape[0]+1e-20), tp / (num_gt+1e-20)]

    def voc_ap(rec, prec):

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    AP = voc_ap(pr_cruve[:,1],pr_cruve[:,0])

    return AP, pr_cruve


def read_box_pred_gt(id, pred_folder, gt_folder,method):
    pred_boxes, level= [], []

    if method == 'Tiny_Face':
        with open(os.path.join(pred_folder,id+'.txt')) as f:
            for line in f.readlines()[2:]:
                line = line.strip().split(' ')
                line_data = [float(line[i]) for i in range(len(line))]
                line_data[2],line_data[3] = line_data[0]+line_data[2],line_data[1]+line_data[3]
                pred_boxes.append(line_data)
        f.close()

    elif method == 'faster_rcnn':
        with open(os.path.join(pred_folder, id+'.json')) as f:
            Info = json.load(f)
        pred_boxes = Info['boxesWithScore']
        pred_boxes = np.array(pred_boxes)
        if len(pred_boxes) > 0:
            pred_boxes = pred_boxes[pred_boxes[:,4]>0.5]

    elif  method == 'LDC_Net':
        with open(os.path.join(pred_folder, id+'.json')) as f:
            Info = json.load(f)
        pred_boxes = Info['boxesWithScore']
        pred_boxes = np.array(pred_boxes)
    else:
        raise ValueError

    with open(os.path.join(gt_folder, id + '.json')) as F:
        Info = json.load(F)
        gt_boxes = np.array(Info['boxes'])
    F.close()

    if len(gt_boxes) > 0:
        w, h = gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1]
        area = w * h
        # print(area)
        level_area = np.zeros(area.shape[0], dtype=np.int32)

        assert area.all() > 0
        for i in range(6):
            lower = (area >= 10 ** i)
            uper = (area < 10 ** (i + 1))
            index = (lower & uper) if i < 5 else lower
            level_area[index] = i
        gt_data = {'num': len(gt_boxes),  'level': level_area, 'boxes':gt_boxes}
    else:
        gt_data = {'num': 0, 'points': np.array([]), 'sigma': [], 'level': np.array([]), 'boxes':[]}
    if len(pred_boxes)>0:
        pred_data= {'boxes':np.array(pred_boxes)}
    else:
        pred_data = {'boxes': np.array([])}



    return pred_data, gt_data

if __name__ == '__main__':
    dataset = 'NWPU' #NWPU  FDST
    method = 'LDC_Net' # Tiny_Face, LDC_Net
    mode = 'val'
    pr_curve_name = './det_pred/'+dataset+'_'+ method+'.json'

    iou_threshold = 0.3
    num_classes = 6
    dataRoot  = '../../ProcessedData/' + dataset

    imageId_txt = os.path.join(dataRoot, mode+'.txt')
    gt_folder = os.path.join(dataRoot, 'jsons')
    if  method == 'Tiny_Face':
        if dataset == 'NWPU':
            pred_folder = './det_pred/Tiny_Face_'+dataset+'/val_results_0.8_0.6'
        if dataset == 'FDST':
            pred_folder = './det_pred/Tiny_Face_' + dataset + '/val_results_0.8_0.3'
    if method == 'faster_rcnn':
        pred_folder = './det_pred/valTestWithScores'
    if method == 'LDC_Net':
        pred_folder = './det_pred/LDC_Net_'+dataset

    with open(imageId_txt) as f:
        img_id = f.readlines()
    f.close()
    matched_mask_score, num_gt = [], 0
    metrics_l = {'tp_c':AverageCategoryMeter(num_classes), 'fn_c':AverageCategoryMeter(num_classes)}

    for i_sample in img_id:
        id = i_sample.strip().split(' ')[0]
        pred_data, gt_data = read_box_pred_gt(id, pred_folder, gt_folder, method)
        print(id)
        mask_score,tp_c, fn_c  = eval_boxes_AP(pred_data['boxes'], gt_data['boxes'], iou_thr=iou_threshold, level=gt_data['level'])

        matched_mask_score.append(mask_score)
        metrics_l['tp_c'].update(tp_c)
        metrics_l['fn_c'].update(fn_c)

        num_gt+=gt_data['num']


    ar_c = metrics_l['tp_c'].sum/(metrics_l['tp_c'].sum+metrics_l['fn_c'].sum+1e-20)

    ap, pr_curve = compute_AP(matched_mask_score, num_gt)
    print(pr_curve)
    save_for_plot = pr_curve[:,[1,0]]
    with open(pr_curve_name,'w') as f:
        print('save pr-curve fpr plotting..............')
        json.dump({'pr_curve': save_for_plot},f,cls=NpEncoder)
    f.close()

    print('-----Detection performance with points annotations-----')
    print('ap: ' + str(ap))
    print('presice@iou0.5: '+ str(pr_curve[-1,0]))
    print('recall@iou0.5: '+ str(pr_curve[-1,1]))
    print('F1@iou0.5: '+ str(2*pr_curve[-1,0]*pr_curve[-1,1]/(pr_curve[-1,0]+pr_curve[-1,1])))

    print('AR_small_category: '+str(ar_c))