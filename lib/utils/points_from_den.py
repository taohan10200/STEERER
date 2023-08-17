import torch
import  torch.nn as nn
import  torch.nn.functional as F

class get_ROI_and_MatchInfo(object):
    def __init__(self,train_size,rdius=8,feature_scale=0.125):
        self.h = train_size[0]
        self.w = train_size[1]
        self.radius = rdius
        self.feature_scale = feature_scale
    def __call__(self,target_a, target_b,noise=None, shape =None):
        gt_a, gt_b = target_a['points'], target_b['points']
        if shape is not None:
            self.h = shape[0]
            self.w = shape[1]
        if noise == 'ab':
            gt_a, gt_b = gt_a + torch.randn(gt_a.size()).to(gt_a)*2, gt_b + torch.randn(gt_b.size()).to(gt_b)*2
        elif noise == 'a':
            gt_a = gt_a + torch.randn(gt_a.size()).to(gt_a)
        elif noise == 'b':
            gt_b = gt_b + torch.randn(gt_b.size()).to(gt_b)


        roi_a = torch.zeros(gt_a.size(0), 5).to(gt_a)
        roi_b = torch.zeros(gt_b.size(0), 5).to(gt_b)
        roi_a[:, 0] = 0
        roi_a[:, 1] = torch.clamp(gt_a[:, 0] - self.radius,min=0)
        roi_a[:, 2] = torch.clamp(gt_a[:, 1] - self.radius, min=0)
        roi_a[:, 3] = torch.clamp(gt_a[:, 0] + self.radius, max=self.w)
        roi_a[:, 4] = torch.clamp(gt_a[:, 1] + self.radius, max=self.h)

        roi_b[:, 0] = 1
        roi_b[:, 1] = torch.clamp(gt_b[:, 0] - self.radius, min=0)
        roi_b[:, 2] = torch.clamp(gt_b[:, 1] - self.radius, min=0)
        roi_b[:, 3] = torch.clamp(gt_b[:, 0] + self.radius, max=self.w)
        roi_b[:, 4] = torch.clamp(gt_b[:, 1] + self.radius, max=self.h)

        pois = torch.cat([roi_a, roi_b], dim=0)

        # ===================match the id for the prediction points of two adhesive frame===================

        a_ids = target_a['person_id']
        b_ids = target_b['person_id']

        dis = a_ids.unsqueeze(1).expand(-1,len(b_ids)) - b_ids.unsqueeze(0).expand(len(a_ids),-1)
        dis = dis.abs()
        matched_a, matched_b = torch.where(dis==0)
        matched_a2b = torch.stack([matched_a,matched_b],1)
        unmatched0 = torch.where(dis.min(1)[0]>0)[0]
        unmatched1 = torch.where(dis.min(0)[0]>0)[0]

        match_gt={'a2b': matched_a2b, 'un_a':unmatched0, 'un_b':unmatched1}

        return  match_gt, pois


def local_maximum_points(density_map, gaussian_maximum,radius=8.,patch_size=128, den_scale=1., threshold=0.15):
    density_map = density_map.detach()
    _,_,h,w = density_map.size()
    # kernel = torch.ones(3,3)/9.
    # kernel =kernel.unsqueeze(0).unsqueeze(0).cuda()
    # weight = nn.Parameter(data=kernel, requires_grad=False)
    # density_map = F.conv2d(density_map, weight, stride=1, padding=1)


    # import pdb
    # pdb.set_trace()
    if h % patch_size != 0:
        pad_dims = (0, 0, 0, patch_size - h % patch_size)
        h = (h // patch_size + 1) * patch_size
        density_map = F.pad(density_map, pad_dims, "constant")


    if w % patch_size != 0:
        pad_dims = (0, patch_size - w % patch_size, 0, 0)
        w = (w // patch_size + 1) * patch_size
        density_map = F.pad(density_map, pad_dims, "constant")


    local_max = F.max_pool2d(density_map, (patch_size, patch_size), stride=patch_size)
    local_max = local_max*threshold
    local_max[local_max<threshold*gaussian_maximum] = threshold*gaussian_maximum
    local_max[local_max>0.3*gaussian_maximum] = 0.3*gaussian_maximum

    local_max = F.interpolate(local_max, scale_factor=patch_size)

    keep = F.max_pool2d(density_map, (3, 3), stride=1, padding=1)
    # keep = F.interpolate(keep, scale_factor=2)
    keep = (keep == density_map).float()
    density_map = keep * density_map

    density_map[density_map < local_max] = 0
    density_map[density_map > 0] = 1
    count = int(torch.sum(density_map).item())

    points = torch.nonzero(density_map)[:,[0,1,3,2]].float() # b,c,h,w->b,c,w,h
    rois = torch.zeros((points.size(0), 5)).float().to(density_map)
    rois[:, 0] = points[:, 0]
    rois[:, 1] = torch.clamp(points[:, 2] - radius, min=0)
    rois[:, 2] = torch.clamp(points[:, 3] - radius, min=0)
    rois[:, 3] = torch.clamp(points[:, 2] + radius, max=w)
    rois[:, 4] = torch.clamp(points[:, 3] + radius, max=h)

    pre_data = {'num': count, 'points': points[:,2:].cpu().numpy()*den_scale, 'rois': rois.cpu().numpy()}
    return pre_data


def object_localization(density_map, gaussian_maximum,patch_size=128,  threshold=0.15):
    density_map = density_map.detach()
    _,_,h,w = density_map.size()

    if h % patch_size != 0:
        pad_dims = (0, 0, 0, patch_size - h % patch_size)
        h = (h // patch_size + 1) * patch_size
        density_map = F.pad(density_map, pad_dims, "constant")

    if w % patch_size != 0:
        pad_dims = (0, patch_size - w % patch_size, 0, 0)
        w = (w // patch_size + 1) * patch_size
        density_map = F.pad(density_map, pad_dims, "constant")

    region_maximum = F.max_pool2d(density_map, (patch_size, patch_size), stride=patch_size)
    region_maximum = region_maximum*threshold
    region_maximum[region_maximum<threshold*gaussian_maximum] = threshold*gaussian_maximum
    region_maximum[region_maximum>0.3*gaussian_maximum] = 0.3*gaussian_maximum

    region_maximum = F.interpolate(region_maximum, scale_factor=patch_size)
    local_maximum = F.max_pool2d(density_map, (3, 3), stride=1, padding=1)
    local_maximum = (local_maximum == density_map).float()
    local_maximum = local_maximum * density_map

    local_maximum[local_maximum < region_maximum] = 0
    local_maximum[local_maximum > 0] = 1

    count = int(torch.sum(local_maximum).item())
    points = torch.nonzero(density_map)[:,[0,1,3,2]].float() # b,c,h,w->b,c,w,h

    pre_data = {'num': count, 'points': points[:,2:].cpu().numpy()}
    return pre_data