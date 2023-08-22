import torch
import torch.nn as nn
from lib.utils.Gaussianlayer import Gaussianlayer
import torch.nn.functional as F
class Counter(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model,
               loss=None, weight=200):
      super(Counter, self).__init__()
      self.model = model
      self.gaussian = Gaussianlayer().cuda()
      self.loss = loss

      self.weight = weight
  def forward(self, inputs, labels, mode='train'):

      outputs = self.model(inputs)
      # outputs = F.upsample(
      #   input=outputs, scale_factor=4, mode='bilinear')
      labels = labels[0].unsqueeze(1)
      labels =  self.gaussian(labels)

      if mode =='train' or mode =='val':
        loss = self.loss(outputs, labels*self.weight)
        gt_cnt = labels.sum().item()
        pre_cnt = outputs.sum().item()/self.weight
        result = {
            'x4': {'gt': gt_cnt, 'error':max(0, gt_cnt-abs(gt_cnt-pre_cnt))},
            'x8': {'gt': 0, 'error': 0},
            'x16': {'gt': 0, 'error': 0},
            'x32': {'gt': 0, 'error': 0}
        }
        return torch.unsqueeze(loss,0), outputs/self.weight, labels, result

      elif mode == 'test':

        return outputs / self.weight

class DrCounter(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model,
               loss=None, weight=200, route_size = (512, 256) ):
      super(DrCounter, self).__init__()
      self.model = model
      self.gaussian = Gaussianlayer().cuda()
      self.loss = loss

      self.weight = weight
      self.route_size = route_size
      self.extern_count = 0
  def forward(self, inputs, labels, mode='train'):
      def vis(data, path='./exp/debug.png'):
          import cv2
          import numpy as np
          data = data.squeeze().cpu().numpy()
          pred_color_map = cv2.applyColorMap((255 * data / (data.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
          cv2.imwrite(path, pred_color_map)

      self.extern_count +=1
      out_list = self.model(inputs)
      B_num, C_num, H_num, W_num =  out_list[0].size()

      label_list = []
      for label in labels:
          label_list.append( self.gaussian(label.unsqueeze(1))*self.weight)
      errorInslice_list = []

      patch_size = self.route_size  # height, width

      patch_h, patch_w = H_num // patch_size[0], W_num // patch_size[1]

      for i, (pre, gt) in enumerate(zip(out_list, label_list)):
          kernel = (int(patch_size[0]/(2**i)), int(patch_size[1]/(2**i)))
          pre_slice = F.unfold(pre, kernel, stride=kernel)
          gt_slice  = F.unfold(gt,  kernel, stride=kernel)

          error_inslice = ((pre_slice-gt_slice).abs()*(gt_slice>0)).sum(1,keepdim=True)
          error_inslice /=((gt_slice>0).sum(1,keepdim=True)+0.1)

          errorInslice_list.append(error_inslice)
      mask_idx = torch.cat(errorInslice_list, dim=1)

      mask_idx = mask_idx.argmin(dim=1, keepdim=True)
      
      import pdb
      pdb.set_trace()
      
      mask = torch.zeros_like(mask_idx).repeat(1,len(out_list),1)
      mask.scatter_(1,mask_idx, 1)
      mask = mask.view(mask.size(0),mask.size(1),patch_h, patch_w).float()
      loss_list = []
      outputs = torch.zeros_like(out_list[0])
      labels = torch.zeros_like(label_list[0])

      result = {}
      for i in range(mask.size(1)):
        guide_mask = F.upsample_nearest(mask[:,i].unsqueeze(1), size=(out_list[i].size()[2:]))

        loss_list.append((((label_list[i]-out_list[i])*guide_mask)**2).sum()/(guide_mask.sum()+0.01))
        gt_cnt = (label_list[i]*guide_mask).sum().item()/self.weight
        pre_cnt = (out_list[i]*guide_mask).sum().item()/self.weight
        result.update({f"x{2**(i+2)}": {'gt': gt_cnt,
                                        'error':max(0, gt_cnt-abs(gt_cnt-pre_cnt))}})

        kernel = (int(patch_size[0] / (2 ** i)), int(patch_size[1] / (2 ** i)))
        if i == 0:
            outputs += (out_list[0] * guide_mask)
            labels += (label_list[0] * guide_mask)
            # print(labels.sum()/200)
            outputs = F.unfold(outputs, kernel,stride=kernel)
            labels = F.unfold(labels, kernel,  stride=kernel)

            B_, KK_, L_ = outputs.size()
            outputs = outputs.transpose(2,1).view(B_ ,L_, kernel[0], kernel[1])
            labels = labels.transpose(2, 1).view(B_, L_, kernel[0], kernel[1])
        else:
            pre_slice = F.unfold(out_list[i],  kernel,stride=kernel)
            gt_slice  = F.unfold(label_list[i], kernel,stride=kernel)
            slice_idx = (mask_idx == i)
            B, KK, L = pre_slice.size()

            pre_slice = pre_slice.transpose(2,1).view(B, L, kernel[0], kernel[1])
            gt_slice = gt_slice.transpose(2,1).view(B, L,kernel[0], kernel[1])
            pad_w, pad_h =(patch_size[1] - kernel[1])//2, (patch_size[0] - kernel[0])//2
            pre_slice = F.pad(pre_slice, [pad_w,pad_w,pad_h,pad_h],  "constant", 0.2)
            gt_slice = F.pad(gt_slice, [pad_w,pad_w,pad_h,pad_h], "constant", 0.2)

            slice_idx = slice_idx.transpose(1, 2).unsqueeze(3)
            pre_slice = (pre_slice*slice_idx)
            gt_slice = (gt_slice * slice_idx)
            outputs += pre_slice
            labels += gt_slice
      # import pdb
      # pdb.set_trace()
      outputs=outputs.view(B_num, patch_h*patch_w, -1).transpose(1,2)
      labels =labels.view(B_num, patch_h*patch_w, -1).transpose(1,2)
      outputs = F.fold(outputs, output_size=(H_num, W_num), kernel_size=patch_size, stride=patch_size)
      labels = F.fold(labels, output_size=(H_num, W_num), kernel_size=patch_size, stride=patch_size)

      if mode =='train' or mode =='val':
        # loss = torch.zeros()
        loss = 1/2*loss_list[0] + 1/4*loss_list[1]+ 1/8*loss_list[2] + 1/16*loss_list[3]

        return torch.unsqueeze(loss,0), outputs/self.weight, labels/self.weight, result

      elif mode == 'test':

        return outputs / self.weight