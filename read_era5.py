# import Ngl, Nio
# ds = Nio.open_file("E:\era5/2012/2012_01_01.grib","r")
import pygrib as pg
# grbs = pg.open("/nvme/era5/2000/2000_01_01.grib")# 所有变量
# for grb in grbs:
# 	print(grb) #每一个变量的头文件
# 	print(grb.keys())  #每一个变量的keys
# 	print (grb.values)  # 每一个变量的值
# 	import pdb
# 	pdb.set_trace()

import torch
path = './exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-09-21-07_reference/Ep_480_mae_68.43758959522378_mse_318.1999576811552.pth'
read_stadict =torch.load(path)
for k,v in read_stadict.items():
	print(k,v)

# path = './exp/NWPU/MocHRBackbone_hrnet48/NWPU_mocHR_small_2022-09-24-20-32/Ep_151_mae_86.99071031267444_mse_326.0298247847295.pth'
# read_stadict =torch.load(path)
# for k,v in read_stadict.items():
# 	print(k,v.size())
import pdb
pdb.set_trace()