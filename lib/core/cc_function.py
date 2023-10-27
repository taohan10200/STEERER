# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from lib.utils.utils import *
from lib.utils.points_from_den import local_maximum_points
from lib.eval.eval_loc_count import eval_loc_MLE_point, eval_loc_F1_boxes

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
def allreduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return  None
    dist.all_reduce(inp,op=dist.ReduceOp.SUM)

def train(config, epoch, num_epoch, epoch_iters, num_iters,
         trainloader, optimizer,scheduler, model, writer_dict, device,img_vis_dir,mean,std,task_KPI,train_dataset):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, label, size, name_idx = batch
        images = images.to(device)
        for i in range(len(label)):
            label[i] = label[i].to(device)

        result = model(images, label, 'train')
        losses=result['losses']
        # import pdb
        # pdb.set_trace()
        pre_den=result['pre_den']['1']
        gt_den = result['gt_den']['1']

        for i in range(len(name_idx[0])):

            _name  = name_idx[0][i]

            if _name not in train_dataset.resize_memory_pool.keys():
                p_h= int(np.ceil(size[i][0]/config.train.route_size[0]))
                p_w = int(np.ceil(size[i][1]/config.train.route_size[1]))
                train_dataset.resize_memory_pool.update({_name:{"avg_size":np.ones((p_h,p_w)),
                                                      "load_num":  np.zeros((p_h,p_w)),
                                                       'size':np.array(size)}})

        loss = losses.mean()


        model.zero_grad()
        loss.backward()
        optimizer.step()

        task_KPI.add({
                'acc1': {'gt': result['acc1']['gt'], 'error': result['acc1']['error']},
                'x4': {'gt':result['x4']['gt'], 'error':result['x4']['error']},
              'x8': {'gt': result['x8']['gt'], 'error': result['x8']['error']},
              'x16': {'gt': result['x16']['gt'], 'error': result['x16']['error']},
              'x32': {'gt': result['x32']['gt'], 'error': result['x32']['error']}

                      })

        KPI = task_KPI.query()
        reduced_loss = reduce_tensor(loss)
        x4_acc = reduce_tensor(KPI['x4']) / world_size
        x8_acc = reduce_tensor(KPI['x8']) / world_size
        x16_acc = reduce_tensor(KPI['x16']) / world_size
        x32_acc = reduce_tensor(KPI['x32']) / world_size
        acc1 = reduce_tensor(KPI['acc1']) / world_size
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        avg_loss.update(reduced_loss.item())
        #
        scheduler.step_update(epoch * epoch_iters + i_iter)

        lr = optimizer.param_groups[0]['lr']
        gt_cnt, pred_cnt = label[0].sum().item() , pre_den.sum().item()
        if i_iter % config.print_freq == 0 and rank == 0:
            print_loss = avg_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.4f}, Loss: {:.4f}, pre: {:.1f}, gt: {:.1f},' \
                  'acc:{:.2f}, accx8:{:.2f},  accx16:{:.2f},accx32:{:.2f},acc1:{:.2f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr*1e5, print_loss,
                pred_cnt,gt_cnt,
                x4_acc.item(), x8_acc.item(), x16_acc.item(), x32_acc.item(),acc1.item())
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            global_steps =  writer_dict['train_global_steps']
            writer_dict['train_global_steps'] = global_steps + 1
            image = images[0]

            if i_iter % 20*config.print_freq == 0:
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)

                save_results_more(global_steps, img_vis_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                  pre_den[0].sum().item(), label[0][0].sum().item())

def validate(config, testloader, model, writer_dict, device,
             num_patches,img_vis_dir,mean,std):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    avg_loss = AverageMeter()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(),
                  'nae': AverageMeter(),'acc1':AverageMeter()}
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # if _>100:
            #     break
            image, label, _, name = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)
            # result = model(image, label, 'val')
            result = patch_forward(model, image, label, num_patches,'val')


            losses=result['losses']
            pre_den=result['pre_den']['1']
            gt_den = result['gt_den']['1']

            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum(), pre_den.sum()

            # print(f" rank: {rank} gt: {gt_count} pre: {pred_cnt}")
            s_mae = torch.abs(gt_count - pred_cnt)

            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))

            allreduce_tensor(s_mae)
            allreduce_tensor(s_mse)
            # acc1 = reduce_tensor(result['acc1']['error']/(result['acc1']['gt']+1e-10))
            reduced_loss = reduce_tensor(losses)
            # print(f" rank: {rank} mae: {s_mae} mse: {s_mse}"
            #       f"loss: {reduced_loss}")
            avg_loss.update(reduced_loss.item())
            # cnt_errors['acc1'].update(acc1)
            cnt_errors['mae'].update(s_mae.item())
            cnt_errors['mse'].update(s_mse.item())

            s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count+1e-10))
            allreduce_tensor(s_nae)
            cnt_errors['nae'].update(s_nae.item())

            if rank == 0:
                if idx % 20==0:
                    # acc1 = cnt_errors['acc1'].avg/world_size
                    # print( f'acc1:{acc1}')
                    image = image[0]
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)
                    save_results_more(name[0], img_vis_dir, image.cpu().data, \
                              pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                      pred_cnt.item(),gt_count.item())
    print_loss = avg_loss.average()/world_size

    mae = cnt_errors['mae'].avg / world_size
    mse = np.sqrt(cnt_errors['mse'].avg / world_size)
    nae = cnt_errors['nae'].avg / world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mae', mae, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return print_loss, mae, mse, nae


def patch_forward(model, img, dot_map, num_patches,mode):
    # crop the img and gt_map with a max stride on x and y axis
    # size: HW: __C_NWPU.TRAIN_SIZE
    # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
    crop_imgs = []
    crop_dots, crop_masks = {},{}

    crop_dots['1'],crop_dots['2'],crop_dots['4'],crop_dots['8'] = [],[],[],[]
    crop_masks['1'],crop_masks['2'],crop_masks['4'],crop_masks['8'] = [], [], [],[]
    b, c, h, w = img.shape
    rh, rw = 768, 1024

    # support for multi-scale patch forward
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

            crop_imgs.append(img[:, :, gis:gie, gjs:gje])
            for res_i in range(len(dot_map)):
                gis_,gie_ = gis//2**res_i, gie//2**res_i
                gjs_,gje_ = gjs//2**res_i, gje//2**res_i
                crop_dots[str(2**res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                mask = torch.zeros_like(dot_map[res_i]).cpu()
                mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                crop_masks[str(2**res_i)].append(mask)

    crop_imgs = torch.cat(crop_imgs, dim=0)
    for k,v in crop_dots.items():
        crop_dots[k] =  torch.cat(v, dim=0)
    for k,v in crop_masks.items():
        crop_masks[k] =  torch.cat(v, dim=0)

    # forward may need repeatng
    crop_losses = []
    crop_preds = {}
    crop_labels = {}
    crop_labels['1'],crop_labels['2'],crop_labels['4'],crop_labels['8'] = [],[],[],[]
    crop_preds['1'],crop_preds['2'],crop_preds['4'],crop_preds['8'] = [], [], [],[]
    nz, bz = crop_imgs.size(0), num_patches
    keys_pre = None

    for i in range(0, nz, bz):
        gs, gt = i, min(nz, i + bz)
        result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys() ],
                                          mode)
        crop_pred = result['pre_den']
        crop_label =  result['gt_den']

        keys_pre = result['pre_den'].keys()
        for k in keys_pre:
            crop_preds[k].append(crop_pred[k].cpu())
            crop_labels[k].append(crop_label[k].cpu())

        crop_losses.append(result['losses'].mean())

    for k in keys_pre:
        crop_preds[k] =  torch.cat(crop_preds[k], dim=0)
        crop_labels[k] =  torch.cat(crop_labels[k], dim=0)


    # splice them to the original size

    result = {'pre_den': {},'gt_den':{}}

    for res_i, k in enumerate(keys_pre):

        pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
        labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
        idx =0
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                gis_,gie_ = gis//2**res_i, gie//2**res_i
                gjs_,gje_ = gjs//2**res_i, gje//2**res_i

                pred_map[:,:, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                labels[:,:, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                idx += 1
        # import pdb
        # pdb.set_trace()
        # for the overlapping area, compute average value
        mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
        pred_map = (pred_map / mask)
        labels = (labels / mask)
        result['pre_den'].update({k: pred_map} )
        result['gt_den'].update({k: labels} )
        result.update({'losses': crop_losses[0]} )
    return result


def test_cc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None):

    model.eval()
    save_count_txt = ''
    device = torch.cuda.current_device()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch

            image, label, _, name = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)


            result = model(image, label, 'val')

            # result = patch_forward(model, image, label,
            #                                       config.test.patch_batch_size, mode='val')

            losses=result['losses']
            pre_den=result['pre_den']['1']
            gt_den = result['gt_den']['1']
            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item() #pre_data['num'] #

            save_count_txt+='{} {}\n'.format(name[0], pred_cnt)
            # import pdb
            # pdb.set_trace()
            msg = '{} {}' .format(gt_count,pred_cnt)
            logger.info(msg)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)


            image = image[0]
            if sv_pred:
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)
                save_results_more(name, sv_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),pred_cnt,gt_count,
                                 )

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mae = cnt_errors['mae'].avg
                mse = np.sqrt(cnt_errors['mse'].avg)
                nae = cnt_errors['nae'].avg
                msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                       nae: {: 4.4f}, Class IoU: '.format(mae,
                                                          mse, nae)
                logging.info(msg)
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

    return  mae, mse, nae,save_count_txt

def test_loc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None,loc_gt=None):

    model.eval()
    device = torch.cuda.current_device()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    num_classes = 6
    max_dist_thresh = 100
    metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}

    loc_100_metrics = {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}

    MLE_metric = AverageMeter()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, size_factor, name = batch
            # if name[0] != '1202':
            #     continue

            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            b, c, h, w = image.size()

            result = model(image, label, 'val')
            # result = patch_forward(model, image, label,
            #                        config.test.patch_batch_size, mode='val')
            # import pdb
            # pdb.set_trace()

            losses=result['losses']
            pre_den=result['pre_den']['1']
            # pre_den_x2=result['pre_den']['2']
            pre_den_x4=result['pre_den']['4']
            pre_den_x8=result['pre_den']['8']

            gt_den = result['gt_den']['1']
            # gt_den_x8 = result['gt_den']['8']

            gt_data = loc_gt[int(name[0])]

            pred_data = local_maximum_points(pre_den.detach(),model.gaussian_maximum, patch_size=32,threshold=config.test.loc_threshold)
            # pred_data_x2 = local_maximum_points(pre_den_x2.detach(),model.gaussian_maximum,patch_size=64,den_scale=2)
            pred_data_x4 = local_maximum_points(pre_den_x4.detach(),model.gaussian_maximum,patch_size=32,den_scale=4,threshold=config.test.loc_threshold)
            pred_data_x8 = local_maximum_points(pre_den_x8.detach(),model.gaussian_maximum,patch_size=16,den_scale=8,threshold=config.test.loc_threshold)

            def nms4points(pred_data, pred_data_x8, threshold):
                points = torch.from_numpy(pred_data['points']).unsqueeze(0)
                points_x8 =  torch.from_numpy(pred_data_x8['points']).unsqueeze(0)


                dist = torch.cdist(points,points_x8)     #torch.Size([1, 16, 16])
                dist = dist.squeeze(0)
                min_val, min_idx = torch.min(dist,0)
                keep_idx_bool = (min_val>threshold)


                keep_idx=torch.where(keep_idx_bool==1)[0]
                if keep_idx.size(0)>0:
                    app_points = (pred_data_x8['points'][keep_idx]).reshape(-1,2)
                    pred_data['points'] = np.concatenate([pred_data['points'], app_points],0)
                    pred_data['num'] =  pred_data['num'] +keep_idx_bool.sum().item()
                return pred_data
            #
            # if name[0] == '3613':
            #     import pdb
            #     pdb.set_trace()
            for idx, down_scale in enumerate([pred_data_x4,pred_data_x8]):
                if pred_data['points'].shape[0]==0 and down_scale['points'].shape[0]>0:
                    pred_data = down_scale
                if pred_data['points'].shape[0]>0  and down_scale['points'].shape[0]>0:
                    pred_data = nms4points(pred_data, down_scale,threshold=(2**(idx+1))*16)

            pred_data_4val  = pred_data.copy()
            pred_data_4val['points'] = pred_data_4val['points']/size_factor.numpy()
            tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_loc_F1_boxes(num_classes, pred_data_4val, gt_data)

            tp_100, fp_100, fn_100 =  0,0,0 #eval_loc_F1_point(pred_data['points'],gt_data['points'],max_dist_thresh = max_dist_thresh)
            Distance_Sum = eval_loc_MLE_point(pred_data['points'], gt_data['points'], 16)

            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item() #
            msg = '{}: gt:{} pre:{}' .format(name, gt_count,pred_cnt)
            logger.info(msg)
            # print(name,':', gt_count, pred_cnt)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)

            MLE_metric.update(Distance_Sum/(gt_data['num']+1e-20), gt_data['num'])

            metrics_l['tp'].update(tp_l)
            metrics_l['fp'].update(fp_l)
            metrics_l['fn'].update(fn_l)
            metrics_l['tp_c'].update(tp_c_l)
            metrics_l['fn_c'].update(fn_c_l)

            metrics_s['tp'].update(tp_s)
            metrics_s['fp'].update(fp_s)
            metrics_s['fn'].update(fn_s)
            metrics_s['tp_c'].update(tp_c_s)
            metrics_s['fn_c'].update(fn_c_s)

            loc_100_metrics['tp_100'].update(tp_100)
            loc_100_metrics['fp_100'].update(fp_100)
            loc_100_metrics['fn_100'].update(fn_100)


            image = image[0]
            if sv_pred:
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)

                save_results_more(name, sv_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),pred_cnt,gt_count,
                                  pred_data['points'],gt_data['points']*size_factor.numpy() )

        # confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        # reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        #
        # confusion_matrix = reduced_confusion_matrix.cpu().numpy()
        # pos = confusion_matrix.sum(1)
        # res = confusion_matrix.sum(0)
        # tp = np.diag(confusion_matrix)
        # IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # mean_IoU = IoU_array.mean()
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mae = cnt_errors['mae'].avg
                mse = np.sqrt(cnt_errors['mse'].avg)
                nae = cnt_errors['nae'].avg
                msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                       nae: {: 4.4f}, Class IoU: '.format(mae,
                                                          mse, nae)
                logging.info(msg)

        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l + 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)

        ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
        ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
        f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s)
        ar_c_s = metrics_s['tp_c'].sum / (metrics_s['tp_c'].sum + metrics_s['fn_c'].sum + 1e-20)

        pre_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum + loc_100_metrics['fp_100'].sum + 1e-20)
        rec_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum + loc_100_metrics['fn_100'].sum + 1e-20)  # True pos rate
        f1_100 = 2 * (pre_100 * rec_100) / (pre_100 + rec_100 + + 1e-20)

        logging.info('-----Localization performance with box annotations-----')
        logging.info('AP_small: '+str(ap_s))
        logging.info('AR_small: '+str(ar_s))
        logging.info('F1m_small: '+str(f1m_s))
        logging.info('AR_small_category: '+str(ar_c_s))
        logging.info('    avg: '+str(ar_c_s.mean()))
        logging.info('AP_large: '+str(ap_l))
        logging.info('AR_large: '+str(ar_l))
        logging.info('F1m_large: '+str(f1m_l))
        logging.info('AR_large_category: '+str(ar_c_l))
        logging.info('    avg: '+str(ar_c_l.mean()))

        logging.info('-----Localization performance with points annotations-----')
        logging.info('avg precision_overall:{}'.format(pre_100.mean()))
        logging.info('avg recall_overall:{}'.format(rec_100.mean()))
        logging.info('avg F1_overall:{}'.format(f1_100.mean()))
        logging.info('Mean Loclization Error:{}'.format(MLE_metric.avg))


        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

        logging.info('-----Counting performance-----')
        logging.info('MAE: ' + str(mae))
        logging.info('MSE: ' + str(mse))
        logging.info('NAE: ' + str(nae))
            # pred = test_dataset.multi_scale_inference(
            #             model,
            #             image,
            #             scales=config.TEST.SCALE_LIST,
            #             flip=config.TEST.FLIP_TEST)
            #

    return  mae, mse, nae

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
