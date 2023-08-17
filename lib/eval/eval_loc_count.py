import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb
import cv2
from .utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

dataset = 'SHHB'
gt_file = './loc_pred/SHHB_test_gt_loc.txt'
# pred_file = 'tiny_val_loc_0.8_0.3.txt'
pred_file = './loc_pred/lsc_cnn_shhb_768x1024.txt' #'./loc_pred/lsc_cnn_shhb_768x1024.txt'
pred_file = './loc_pred/TopoCount_partb.txt'
flagError = False

if dataset == 'NWPU':
    id_std = [i for i in range(3110,3610,1)]  #for nwpu
    id_std[59] = 3098
if dataset == 'SHHB':
    id_std = [i for i in range(401,717,1)] # for shhb
if dataset == 'SHHA':
    id_std = [i for i in range(301,483,1)] # for shha
if dataset == 'QNRF':
    id_std = [i for i in range(1202,1536,1)] # for QNRF

num_classes = 6

def eval_loc_MLE_point(pred_points, gt_points, penalty=16):
    # The predictions are matched to head annotations in a one-to-one fashion and a fixed penalty of
    # 16 pixels is added for absent or spurious detections.

    # this metric is first proposed by the "Locate, Size and Count: Accurately Resolving
    # People in Dense Crowds via Detectio,T-PAMI,2020"
    def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, sigma):

        for i_pred_p in range(pred_num):
            pred_dist = dist_matrix[i_pred_p, :]
            match_matrix[i_pred_p, :] = pred_dist <= penalty

        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]   # belong to ground truth, but cannot be detected
        tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0] # Those be detected but not belong to ground truth


        tp = tp_pred_index.shape[0]
        fp = fp_pred_index.shape[0]
        fn = fn_gt_index.shape[0]

        distance_sum = (dist_matrix[assign]).sum() + max(fp,fn)*penalty

        return distance_sum

    Distance_sum = 0
    # import pdb
    # pdb.set_trace()
    if len(gt_points) == 0 and len(pred_points) != 0:
        Distance_sum = pred_points.shape[0]*penalty

    if len(gt_points) != 0 and len(pred_points) == 0:
        Distance_sum = gt_points.shape[0] * penalty

    if len(gt_points) != 0 and len(pred_points) != 0:
        dist_matrix = ss.distance_matrix(pred_points, gt_points, p=2)
        match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
        Distance_sum  = compute_metrics(dist_matrix, match_matrix, pred_points.shape[0], gt_points.shape[0], penalty)

    return Distance_sum

def eval_loc_F1_point(pred_points, gt_points, max_dist_thresh = 100):
    def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, sigma):
        for i_pred_p in range(pred_num):
            pred_dist = dist_matrix[i_pred_p, :]
            match_matrix[i_pred_p, :] = pred_dist <= sigma

        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
        tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

        tp = tp_pred_index.shape[0]
        fp = fp_pred_index.shape[0]
        fn = fn_gt_index.shape[0]

        return tp, fp, fn

    # the arrays for tp, fp, fn, precision, recall, and f1 only use the entries from 1 to max_dist_thresh. Do not use index 0.
    tp_class = np.zeros(max_dist_thresh )
    fp_class = np.zeros(max_dist_thresh )
    fn_class = np.zeros(max_dist_thresh )

    for dist_thresh in range(0, max_dist_thresh):

        tp, fp, fn = [0, 0, 0]

        if len(gt_points) == 0 and len(pred_points) != 0:
            fp_pred_index = np.array(range(pred_points.shape[0]))
            fp = fp_pred_index.shape[0]

        if len(gt_points) != 0 and len(pred_points) == 0:
            fn_gt_index = np.array(range(gt_points.shape[0]))
            fn = fn_gt_index.shape[0]

        if len(gt_points) != 0 and len(pred_points) != 0:
            dist_matrix = ss.distance_matrix(pred_points, gt_points, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            tp, fp, fn  = compute_metrics(dist_matrix, match_matrix, pred_points.shape[0], gt_points.shape[0], dist_thresh+1)

        # false positive, fp,  remaining points in prediction that were not matched to any point in ground truth
        tp_class[dist_thresh] += tp
        fp_class[dist_thresh] += fp
        fn_class[dist_thresh] += fn

    return tp_class, fp_class, fn_class

def eval_loc_F1_boxes(num_classes, pred_data, gt_data):
    def compute_metrics(dist_matrix,match_matrix,pred_num,gt_num,sigma,level):
        for i_pred_p in range(pred_num):
            pred_dist = dist_matrix[i_pred_p,:]
            match_matrix[i_pred_p,:] = pred_dist<= sigma

        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
        tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
        tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]
        level_list = level[tp_gt_index]

        tp = tp_pred_index.shape[0]
        fp = fp_pred_index.shape[0]
        fn = fn_gt_index.shape[0]

        tp_c = np.zeros([num_classes])
        fn_c = np.zeros([num_classes])

        for i_class in range(num_classes):
            tp_c[i_class] = (level[tp_gt_index]==i_class).sum()
            fn_c[i_class] = (level[fn_gt_index]==i_class).sum()

        return tp,fp,fn,tp_c,fn_c

    tp_s, fp_s, fn_s, tp_l, fp_l, fn_l = [0, 0, 0, 0, 0, 0]
    tp_c_s = np.zeros([num_classes])
    fn_c_s = np.zeros([num_classes])
    tp_c_l = np.zeros([num_classes])
    fn_c_l = np.zeros([num_classes])

    if gt_data['num'] == 0 and pred_data['num'] != 0:
        pred_p = pred_data['points']
        fp_pred_index = np.array(range(pred_p.shape[0]))
        fp_s = fp_pred_index.shape[0]
        fp_l = fp_pred_index.shape[0]

    if pred_data['num'] == 0 and gt_data['num'] != 0:
        gt_p = gt_data['points']
        level = gt_data['level']
        fn_gt_index = np.array(range(gt_p.shape[0]))
        fn_s = fn_gt_index.shape[0]
        fn_l = fn_gt_index.shape[0]
        for i_class in range(num_classes):
            fn_c_s[i_class] = (level[fn_gt_index] == i_class).sum()
            fn_c_l[i_class] = (level[fn_gt_index] == i_class).sum()

    if gt_data['num'] != 0 and pred_data['num'] != 0:
        pred_p = pred_data['points']
        gt_p = gt_data['points']
        sigma_s = gt_data['sigma'][:, 0]
        sigma_l = gt_data['sigma'][:, 1]
        level = gt_data['level']

        # dist
        dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
        match_matrix = np.zeros(dist_matrix.shape, dtype=bool)

        # sigma_s and sigma_l
        tp_s, fp_s, fn_s, tp_c_s, fn_c_s = compute_metrics(dist_matrix, match_matrix, pred_p.shape[0], gt_p.shape[0], sigma_s, level)
        tp_l, fp_l, fn_l, tp_c_l, fn_c_l = compute_metrics(dist_matrix, match_matrix, pred_p.shape[0], gt_p.shape[0], sigma_l, level)
    return tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l

def main():
    
    cnt_errors = {'mae':AverageMeter(),'mse':AverageMeter(),'nae':AverageMeter(),}
    metrics_s = {'tp':AverageMeter(), 'fp':AverageMeter(), 'fn':AverageMeter(), 'tp_c':AverageCategoryMeter(num_classes), 'fn_c':AverageCategoryMeter(num_classes)}
    metrics_l = {'tp':AverageMeter(), 'fp':AverageMeter(), 'fn':AverageMeter(), 'tp_c':AverageCategoryMeter(num_classes), 'fn_c':AverageCategoryMeter(num_classes)}

    MLE_metric = AverageMeter()

    max_dist_thresh = 100
    loc_100_metrics = {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}


    pred_data, gt_data = read_pred_and_gt(pred_file,gt_file)
    for i_sample in id_std:
        print(i_sample) 
        # init

        Distance_Sum = eval_loc_MLE_point(pred_data[i_sample]['points'], gt_data[i_sample]['points'], 16)
        MLE_metric.update(Distance_Sum, gt_data[i_sample]['num'])

        tp_100, fp_100, fn_100 = eval_loc_F1_point(pred_data[i_sample]['points'],gt_data[i_sample]['points'],max_dist_thresh = max_dist_thresh)
        loc_100_metrics['tp_100'].update(tp_100)
        loc_100_metrics['fp_100'].update(fp_100)
        loc_100_metrics['fn_100'].update(fn_100)

        tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l=eval_loc_F1_boxes(num_classes, pred_data[i_sample], gt_data[i_sample])
        metrics_s['tp'].update(tp_s)
        metrics_s['fp'].update(fp_s)
        metrics_s['fn'].update(fn_s)
        metrics_s['tp_c'].update(tp_c_s)
        metrics_s['fn_c'].update(fn_c_s)
        metrics_l['tp'].update(tp_l)
        metrics_l['fp'].update(fp_l)
        metrics_l['fn'].update(fn_l)
        metrics_l['tp_c'].update(tp_c_l)
        metrics_l['fn_c'].update(fn_c_l)

        gt_count,pred_cnt = gt_data[i_sample]['num'],pred_data[i_sample]['num']
        s_mae = abs(gt_count-pred_cnt)
        s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)
        cnt_errors['mae'].update(s_mae)
        cnt_errors['mse'].update(s_mse)

        if gt_count !=0:
            s_nae = abs(gt_count-pred_cnt)/gt_count
            cnt_errors['nae'].update(s_nae)
    
    ap_s = metrics_s['tp'].sum/(metrics_s['tp'].sum+metrics_s['fp'].sum+1e-20)
    ar_s = metrics_s['tp'].sum/(metrics_s['tp'].sum+metrics_s['fn'].sum+1e-20)
    f1m_s = 2*ap_s*ar_s/(ap_s+ar_s)
    ar_c_s = metrics_s['tp_c'].sum/(metrics_s['tp_c'].sum+metrics_s['fn_c'].sum+1e-20)


    ap_l = metrics_l['tp'].sum/(metrics_l['tp'].sum+metrics_l['fp'].sum+1e-20)
    ar_l = metrics_l['tp'].sum/(metrics_l['tp'].sum+metrics_l['fn'].sum+1e-20)
    f1m_l = 2*ap_l*ar_l/(ap_l+ar_l)
    ar_c_l = metrics_l['tp_c'].sum/(metrics_l['tp_c'].sum+metrics_l['fn_c'].sum+1e-20)

    print('-----Localization performance-----')
    print('AP_small: '+str(ap_s))
    print('AR_small: '+str(ar_s))
    print('F1m_small: '+str(f1m_s))
    print('AR_small_category: '+str(ar_c_s))
    print('    avg: '+str(ar_c_s.mean()))
    print('AP_large: '+str(ap_l))
    print('AR_large: '+str(ar_l))
    print('F1m_large: '+str(f1m_l))
    print('AR_large_category: '+str(ar_c_l))
    print('    avg: '+str(ar_c_l.mean()))

    mae = cnt_errors['mae'].avg
    mse = np.sqrt(cnt_errors['mse'].avg)
    nae = cnt_errors['nae'].avg

    print('-----Counting performance-----')
    print('MAE: '+str(mae))
    print('MSE: '+str(mse))
    print('NAE: '+str(nae))

    pre_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum  + loc_100_metrics['fp_100'].sum + 1e-20)
    rec_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum  + loc_100_metrics['fn_100'].sum + 1e-20) # True pos rate
    f1_100 = 2 * (pre_100 * rec_100) / (pre_100 + rec_100++ 1e-20)
    print('-----Localization performance with points annotations-----')
    print('avg precision_overall', pre_100.mean())
    print('avg recall_overall',    rec_100.mean())
    print('avg F1_overall',        f1_100.mean())
    print('Mean Loclization Error', MLE_metric.avg)

if __name__ == '__main__':
    main()
