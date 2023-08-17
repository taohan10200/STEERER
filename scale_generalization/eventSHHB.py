from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val

def slide_filter(val,factor=0.8):
    for i,data in enumerate(val,1):
        if i <len(val):
          
            val[i] =  factor*val[i-1]+(1-factor)*val[i]
    return val

def draw_plt(val0,val1,val2,val3, val_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    # 常用字体: ’Dejavu Sans‘,’Times New Roman‘,'Arial'
    # 设置图片尺寸  set size of fgure units cm
    plt.rcParams['figure.figsize'] = (5.44,3.70)
    plt.xlabel('Validation Steps')
    # plt.ylabel('Loss')
    plt.ylabel('Mean Square Error')
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    # 采用紧凑布局
    plt.tight_layout()
    # 生成数据
    step0,val0=[i.step for i in val0], [j.value for j in val0]
    step1,val1=[i.step for i in val1], [j.value for j in val1]
    step2,val2=[i.step+29 for i in val2], [j.value for j in val2]
    step1 = step1 + step2
    val1 = val1 + val2
    step2,val2=[i.step for i in val3], [j.value for j in val3]

    val0=slide_filter(val0)
    val1=slide_filter(val1)
    val2=slide_filter(val2)
    # val3=slide_filter(val3)

    # line color
    # set 是经典的RGB色，简直辣眼睛
    # set1: red,green,blue
    # set2: frebrick,forestgreen,darkgreen
    # plt.plot(step[0:160],val[0:160],color='firebrick',marker='^',markevery=10,mew=0.25)
    # plt.plot(step1[0:160],val1[0:160],color='forestgreen',marker='o',markevery=10,mew=0.25)
    # plt.plot(step2[0:160],val2[0:160],color='darkblue',marker='s',markevery=10,mew=0.25)
    num = min(len(val0),len(val1),len(val2))
    plt.plot(step0[0:num],val0[0:num],color='#FF7F0E',mew=0.25)
    plt.plot(step1[0:num],val1[0:num],color='#1F77B4',mew=0.25)
    plt.plot(step2[0:num],val2[0:num],color='#2CA02C',mew=0.25)
    # plt.plot(step3[0:160],val3[0:160],color='#FF7F0E',mew=0.25) #ED1C24
    # create legend without frame
    plt.title(val_name)
    # 绘制图例，不要框框
    plt.legend(['BL1_concat','BL2_FPN','STEERER (Ours)'],frameon=False)
    plt.grid(axis='y')
    # plt.xlim(0,150)
    # plt.ylim(0,350)
    ax = plt.gca()  #获取当前图像的坐标轴信息
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))  # Or whatever your limits are . . .
    # gca().yaxis.set_major_formatter(xfmt)
    plt.savefig(val_name+'_SHHB.png',dpi=600,format='png')
    plt.savefig(fname=val_name+'_SHHB.svg',format="svg")
    plt.figure()
    plt.savefig(fname=val_name+'_SHHB.svg',format="svg")
    plt.show()

    # plt.savefig(val_name+'_SHHB.svg',dpi=600,format='svg')



if __name__ == "__main__":

    concat = './exp/NWPU/MocHRBackbone_hrnet48/NWPU_HR_base_concat_2022-11-04-23-00/events.out.tfevents.1667574052.SH-IDC1-10-140-1-114'
    SIL = './exp/NWPU/MocHRBackbone_hrnet48/NWPU_HR_base_2022-11-04-01-01/events.out.tfevents.1667494868.SH-IDC1-10-140-1-31'
    fpn1 = './exp/NWPU/HRBackboneFPN_hrnet48/NWPU_HR_base_fpn_2022-11-03-12-33/events.out.tfevents.1667449987.SH-IDC1-10-140-1-31'
    fpn2 = './exp/NWPU/HRBackboneFPN_hrnet48/NWPU_HR_base_fpn_2022-11-03-12-33/events.out.tfevents.1667545113.SH-IDC1-10-140-1-114'

    val_name = 'valid_mae'
    # val_name = 'data/val_loss'
    val1 = read_tensorboard_data(concat, val_name)

    val20 = read_tensorboard_data(fpn1, val_name)
    val21 = read_tensorboard_data(fpn2, 'valid_mae')
    val3 = read_tensorboard_data(SIL, val_name)

    # val0 = read_tensorboard_data(da_path, 'data/val_loss_NoAdpt')
    draw_plt(val1, val20, val21, val3, 'MAE on NWPU-Crowd Validation Set')
    # draw_plt(val0,val1,val2,val3, 'Validation loss on Shanghai Tech Part B')
