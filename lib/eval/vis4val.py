import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb

import cv2
from utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

gt_file = 'val_gt_loc.txt'



exp_name = './RAZ_results'
pred_file = 'Raz_loc_val_0.5.txt'



img_path = ori_data = '/media/D/DataSet/NWPU-ori/images'

flagError = False
id_std = [i for i in range(3110,3610,1)]
id_std[59] = 3098


if not os.path.exists(exp_name):
    os.mkdir(exp_name)



def main():
    
    pred_data, gt_data = read_pred_and_gt(pred_file,gt_file)


    for i_sample in id_std:

        print(i_sample)        
        
        gt_p,pred_p,fn_gt_index,tp_pred_index,fp_pred_index,ap,ar= [],[],[],[],[],[],[]

        if gt_data[i_sample]['num'] ==0 and pred_data[i_sample]['num'] !=0:            
            pred_p =  pred_data[i_sample]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            ap = 0
            ar = 0

        if pred_data[i_sample]['num'] ==0 and gt_data[i_sample]['num'] !=0:
            gt_p = gt_data[i_sample]['points']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            sigma_l = gt_data[i_sample]['sigma'][:,1]
            ap = 0
            ar = 0

        if gt_data[i_sample]['num'] !=0 and pred_data[i_sample]['num'] !=0:
            pred_p =  pred_data[i_sample]['points']    
            gt_p = gt_data[i_sample]['points']
            sigma_l = gt_data[i_sample]['sigma'][:,1]
            level = gt_data[i_sample]['level']        
        
            # dist
            dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
            match_matrix = np.zeros(dist_matrix.shape,dtype=bool)
            for i_pred_p in range(pred_p.shape[0]):
                pred_dist = dist_matrix[i_pred_p,:]
                match_matrix[i_pred_p,:] = pred_dist<=sigma_l
                
            # hungarian outputs a match result, which may be not optimal. 
            # Nevertheless, the number of tp, fp, tn, fn are same under different match results
            # If you need the optimal result for visualzation, 
            # you may treat it as maximum flow problem. 
            tp, assign = hungarian(match_matrix)
            fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
            tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
            tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
            fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]



            pre = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fp_pred_index.shape[0]+1e-20)
            rec = tp_pred_index.shape[0]/(tp_pred_index.shape[0]+fn_gt_index.shape[0]+1e-20)

        img = cv2.imread(img_path + '/' + str(i_sample) + '.jpg')#bgr
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        
        point_r_value = 5
        thickness = 3
        if gt_data[i_sample]['num'] !=0:
            for i in range(gt_p.shape[0]):
                if i in fn_gt_index:                
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),point_r_value,(0,0,255),-1)# fn: red
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,0,255),thickness)#  
                else:
                    cv2.circle(img,(gt_p[i][0],gt_p[i][1]),sigma_l[i],(0,255,0),thickness)# gt: green
        if pred_data[i_sample]['num'] !=0:
            for i in range(pred_p.shape[0]):
                if i in tp_pred_index:
                    cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value,(0,255,0),-1)# tp: green
                else:                
                    cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value*2,(255,0,255),-1) # fp: Magenta

        cv2.imwrite(exp_name+'/'+str(i_sample)+ '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.jpg', img)

from eval_det import *
def box_vis(method='Tiny_Face' ): # Tiny_Face, LDC_Net
    dataset = 'NWPU'
    dataRoot  = '../../ProcessedData/' + dataset
    saved_img_path = os.path.join('../output', method, 'box_vis')
    if not os.path.exists(saved_img_path):
        os.makedirs(saved_img_path)

    gt_folder = os.path.join(dataRoot, 'jsons')
    if method == 'Tiny_Face':
            pred_folder = './det_pred/Tiny_Face_' + dataset + '/val_results_0.8_0.3'
    if method == 'faster_rcnn':
        pred_folder = './det_pred/valTestWithScores'
    if method == 'LDC_Net':
        pred_folder = './det_pred/LDC_Net_' + dataset



    for id in id_std:
        id = str(id)
        if id not in ['3114','3141','3146','3161','3194','3217','3313','3327','3429','3536','3586']:
            continue

        pred_data, gt_data = read_box_pred_gt(id, pred_folder, gt_folder, method)
        # print(pred_data, gt_data)

        mask_score_list,_,_ = eval_boxes_AP(pred_data['boxes'], gt_data['boxes'], iou_thr=0.5, level=gt_data['level'])
        print(id)

        ori_img_path = os.path.join(dataRoot, 'images', id+'.jpg')

        mask_score = np.array(mask_score_list)
        if mask_score.ndim == 2:
            det_num = mask_score[:, 0].sum()
            pre = mask_score[:, 0].sum() / (mask_score.shape[0] + 1e-20)
            rec = mask_score[:, 0].sum() / (gt_data['num'] + 1e-20)
        else:
            pre = 0
            rec = 0
            det_num = 0
        img = cv2.imread(ori_img_path)  # bgr
        box_img = img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        yellow = (0, 255, 255)  # yellow
        font = cv2.FONT_HERSHEY_SIMPLEX
        spring_green = (69, 139, 0)  # BGR
        green = (0, 255, 0)
        thickness = 2
        lineType = 4

        if gt_data['num'] != 0:
            for i, box in enumerate(gt_data['boxes'], 0):
                wh_LeftTop = (int(box[0]), int(box[1]))
                wh_RightBottom = (int(box[2]), int(box[3]))
                cv2.rectangle(img, wh_LeftTop, wh_RightBottom, spring_green, thickness, lineType)
        cv2.imwrite(saved_img_path + '/' + id + '_gt_box.jpg', img)
        if pred_data['boxes'].shape[0] != 0:
            for i, box in enumerate(pred_data['boxes'], 0):
                wh_LeftTop = (int(box[0]), int(box[1]))
                wh_RightBottom = (int(box[2]), int(box[3]))
                cv2.rectangle(box_img, wh_LeftTop, wh_RightBottom, green, thickness, lineType)
        cv2.imwrite(saved_img_path + '/' + id + '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] +
                    '_gt_' + str(gt_data['num']) + '_det_' + str(mask_score[:, 0].sum()) +
                    '_box_'+str(pred_data['boxes'].shape[0])+method+'.jpg', box_img)



if __name__ == '__main__':
    # main()
    box_vis(method='Tiny_Face' ) # Tiny_Face Tiny_Face, LDC_Net