import random
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque

class Task_KPI_Pool:
    def __init__(self,task_setting, maximum_sample,device):
        """
        :param task_setting: {'den': ['gt', 'den'], 'match': ['gt', 'den']}
        :param maximum_sample: the number of the saved samples
        """
        self.pool_size = maximum_sample
        self.maximum_sample = maximum_sample
        assert self.pool_size > 0
        self.current_sample = {x: 0   for x in task_setting.keys()}
        self.store = task_setting
        for key, data in self.store.items():
            self.store[key] = {x: deque() for x in data}
        self.device = device
    def add(self, save_dict):
        """
        :param save_dict:  {'den': {'gt':torch.tensor(10), 'den':torch.tensor(20)},
                            'match': {'gt':torch.tensor(40), 'den':torch.tensor(100)}}
        :return: None
        """
        for task_key, data in save_dict.items():
            if self.current_sample[task_key]< self.pool_size:
                self.current_sample[task_key] = self.current_sample[task_key] + 1
                for data_key, data_val in data.items():
                    self.store[task_key][data_key].append(data_val)
            else:
                for data_key, data_val in data.items():
                    self.store[task_key][data_key].popleft()
                    self.store[task_key][data_key].append(data_val)

    def return_feature(self,cls_group):
        return_features = []
        return_labels = []

        return  return_features, return_labels

    def query(self):
        task_KPI = {}
        for task_key in self.store:
            data_keys = list(self.store[task_key].keys())

            gt_list = list(self.store[task_key][data_keys[0]])
            correct_list = list(self.store[task_key][data_keys[1]])
            gt_sum = torch.tensor(gt_list).to(self.device).sum()

            correct_sum = torch.tensor(correct_list).to(self.device).sum()


            task_KPI.update({task_key:correct_sum/(gt_sum+1e-8)})

        return  task_KPI

if __name__ == '__main__':
    import random

    index = np.random.randint(0, 3, size=30)
    # index = random.sample(range(0, 54), 54)
    feature = torch.rand(30,3).cuda()
    target = torch.Tensor(index).cuda().long()
    pred = torch.randn(30,3).cuda()
    task = {'den': ['gt', 'den'], 'match': ['gt', 'den']}
    save_dict0 =  {'den': {'gt':torch.tensor(10), 'den':torch.tensor(20)}, 'match': {'gt':torch.tensor(40), 'den':torch.tensor(100)}}
    save_dict1 =  {'den': {'gt':torch.tensor(20.6), 'den':torch.tensor(30.8)}, 'match': {'gt':torch.tensor(50), 'den':torch.tensor(120.4)}}
    print(task.keys())
    pool = Task_KPI_Pool(task,100)
    pool.add(save_dict0)
    pool.add(save_dict1)

    print(pool.query())

    import pdb

    pdb.set_trace()