import torch
# from config import cfg
from models.mtl_ import MtlLearner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# model =  MtlLearner(cfg, mode='pre', backbone=cfg.model_type)

# # the histogram of the data
# # 50：将数据分成50组
# # facecolor：颜色；alpha：透明度
# # density：是密度而不是具体数值
def hist(sou,tar):
    plt.figure(1)
    n1, bins1, patches1 = plt.hist(sou, 30, density=True, facecolor='b',histtype='stepfilled', alpha=1,label='Supervised with Synthetic Data')
    n2, bins2, patches2 = plt.hist(tar, 30, density=True, facecolor='r',histtype='stepfilled', alpha=0.2,label='Supervised with Real-World Data')
    # n：概率值；bins：具体数值；patches：直方图对象。

    plt.xlabel('Bias')
    plt.ylabel('Probability(%)')
    plt.title('Distribution of Bias in VGG Feature.12')

    plt.text(-0.16, 12, r'$\mu=0.040,\ \sigma=0.006$')
    plt.text(-0.18, 3, r'$\mu=0.028,\ \sigma=0.003$')

    # 设置x，y轴的具体范围
    # plt.axis([np.min(sou), np.max(sou), np.min(np.min(sou),np.min(tar)), np.max(np.max(sou),np.max(tar))])
    plt.legend()
    plt.grid(True)
    plt.show()

def pad(sou=None,tar=None):
    means = 10, 20
    stdevs = 4, 2
    dist = pd.DataFrame(
    np.vstack((sou,tar)).transpose(1,0),
    columns = ['Supervised with synthetic data', 'Supervised with real-world Data'])
    dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)
    fig, ax = plt.subplots()
    dist.plot.kde(ax=ax, legend=False,color=('b','r'), title='Distribution of the vgg feature.12.bias',fontsize=10)
    dist.plot.hist(density=True, color=('b','r'),alpha=0.6,ax=ax,fontsize=10)
    ax.set_ylabel('Probability',fontsize=10)
    ax.set_xlabel('bias',fontsize=10)
    ax.grid(axis='y')
    ax.text(-0.4, 9.8, r'$\mu=4.0e-3,\ \sigma=5.6e-4$',fontsize=10)
    ax.text(-0.4, 9.2,  r'$\mu= 2.8e-3,\ \sigma=3.4e-4$',fontsize=10)
    # ax.set_facecolor('#d8dcd6')
    fig.show()
    fig.savefig('bias_avg.png',dpi=600,format='png')

def pad1(sou=None,tar=None):
    means = 10, 20
    stdevs = 4, 2
    dist = pd.DataFrame(
    np.vstack((sou,tar)).transpose(1,0),
    columns = ['Supervised with synthetic data', 'Supervised with real-world Data'])
    dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)
    fig, ax = plt.subplots()
    dist.plot.kde(ax=ax, legend=False,color=('b','r'), title='Distribution of the means in vgg feature.12 Conv kernel',fontsize=10)
    dist.plot.hist(bins=20,density=True, color=('b','r'),alpha=0.5,ax=ax,fontsize=10)
    ax.set_ylabel('Probability',fontsize=10)
    ax.set_xlabel('the means of Conv kernel',fontsize=10)
    ax.grid(axis='y')
    ax.text(-0.093, 74.5, r'$\mu=-8.5e-5,\ \sigma=1.5e-5$',fontsize=10)
    ax.text(-0.093, 70,  r'$\mu= 3.8e-5,\ \sigma=5.6e-5$',fontsize=10)
    # ax.set_facecolor('#d8dcd6')
    ax.set_xlim([-0.1,0.1]) 
    fig.show()
    fig.savefig('bw_avg.png',dpi=600,format='png')
if __name__ == '__main__':

    sou_weights = '/media/D/ht/cross_scene_crowd_counting/exp/pre/all_ep_23_mae_48.7_mse_117.6.pth'
    tar_weights = '/media/D/ht/cross_scene_crowd_counting/exp/pre/all_ep_43_mae_18.8_mse_40.0.pth'
    # model.load_state_dict(torch.load(nit_weights))

    sou_dict = torch.load(sou_weights)
    tar_dict = torch.load(tar_weights)

    # for i,(name, para) in enumerate(model.named_parameters(),1):
    #     if i==1:
    #         print(name,para)

    #     if 'mtl' in name:
    #         para.requires_grad=False
    #         print(name, para)

    # para.requires_grad = False
    # for _ ,(i,b) in enumerate(model.state_dict().items(),1):
    #     if _==22:
    #         print(i,b)
    # summary(model,(3,480,480), batch_size=1)
    #
    for _, (i, b) in enumerate(sou_dict.items(), 1):
        if _ == 11:
            print(b.shape)
            sou_w = torch.mean(b.view(256, 256, -1), 2).view(-1).cpu().data.numpy()
            
        if _==12:
            sou= b.cpu().data.numpy()
            # print(i,b)
            break

    for _, (i, b) in enumerate(tar_dict.items(), 1):
        if _ == 11:
            tar_w = torch.mean(b.view(256, 256, -1), 2).view(-1).cpu().data.numpy()
    
        if _==12:
            tar= b.cpu().data.numpy()
            # print(i,b)
            break

    # pad(sou,tar)
    pad1(sou_w,tar_w)
    mu1, sigma1 = np.mean(sou), np.var(sou)
    mu2, sigma2 = np.mean(tar), np.var(tar)
    print(mu1, sigma1)
    print(mu2, sigma2)