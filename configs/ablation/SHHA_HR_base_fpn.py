# _base_ =[
#     '../_base_/datasets/imagenet_bs64_swin_224.py'
# ]
gpus = (0, 1,)
log_dir = 'exp'
workers = 12
print_freq = 30
seed = 3035

network = dict(
    backbone="HRBackboneFPN",
    sub_arch='hrnet48',
    counter_type = 'single_resolution', #'withMOE' 'baseline'
    resolution_num = [0,1,2,3],
    loss_weight = [1., 1/2, 1/4, 1/8],
    sigma = [4],
    gau_kernel_size = 15,
    baseline_loss = False,
    pretrained_backbone="../PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth",
    # '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/imagenet/CCFBackbone_hrt_base/ccformer_b_2022-08-16-14-59/Ep_142_acc1_75.12800201416016.pth',

    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=128*4,
        stages_channel = [384, 192, 96, 48],
        inter_layer=[64,32,16],
        out_channels=1)
    )

dataset = dict(
    name='SHHA',
    root='../ProcessedData/SHHA/',
    test_set='test_val.txt',
    train_set='train.txt',
    loc_gt = 'test_gt_loc.txt',
    num_classes=len(network['resolution_num']),
    den_factor=100,
    extra_train_set =None
)


optimizer = dict(
    NAME='adamw',
    BASE_LR=1e-5,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-2,
    EPS= 1.0e-08,
    MOMENTUM= 0.9,
    AMSGRAD = False,
    NESTEROV= True,
    )


lr_config = dict(
    NAME='cosine',
    WARMUP_METHOD='linear',
    DECAY_EPOCHS=250,
    DECAY_RATE = 0.1,
    WARMUP_EPOCHS=10,   # the number of epochs to warmup the lr_rate
    WARMUP_LR=5.0e-07,
    MIN_LR= 1.0e-07
  )

total_epochs = 210

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

train = dict(
    counter='normal',
    image_size=(768, 768),  # height width
    route_size=(256, 256),  # height, width
    base_size=2048,
    batch_size_per_gpu=6,
    shuffle=True,
    begin_epoch=0,
    end_epoch=1000,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path=None, #'./exp/SHHA/MocHRBackbone_hrnet48/SHHA_mocHR_small_2022-09-26-20-30', #'/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/SHHB/MocHRBackbone_hrnet48/mocHR_small_2022-09-18-14-54/', #'/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-13-16-28/', # '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-09-21-07/', #'./exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-06-20-35/', #'/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-06-13-03/', #'./exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-04-19-21/', #'./exp/NWPU/MocBackbone_moc_small/moc_small_2022-08-31-14-41/', #'./exp/NWPU/seg_hrnet/hrt_small_2022-08-07-23-15/',
    # './exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-05-27-15-03'
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_span = [-1000, -600, -400, -200, -200, -100, -100],
    downsamplerate= 1,
    ignore_label= 255
)


test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=2048,
    loc_base_size=2048,
    loc_threshold = 0.2,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,

    # model_file= './exp/SHHA/MocHRBackbone_hrnet48/SHHA_mocHR_small_2022-09-22-20-47_baseline/Ep_319_mae_57.58954155052101_mse_96.65610938266916.pth'
    # model_file= './exp/SHHA/MocCatBackbone_hrnet48/SHHA_catHR_small_2022-09-27-16-47_best/Ep_671_mae_54.692220457307585_mse_90.16555071890785.pth'
    model_file= './exp/SHHA/HRBackboneFPN_hrnet48/SHHA_HR_base_fpn_2022-11-01-14-02/Ep_359_mae_61.589104935363096_mse_99.79066426707645.pth'   #final_state.pth'
    #cross dataset valization
    # model_file = './exp/NWPU/HRBackboneFPN_hrnet48/NWPU_HR_base_fpn_2022-11-03-12-33/Ep_109_mae_52.69846382488031_mse_122.5832399332522.pth'
    # mae:  77.0864, mse:  142.9153,                nae:  0.1681, Class IoU:
    # model_file= './exp/QNRF/HRBackboneFPN_hrnet48/QNRF_HR_4096_2048_fpn_2022-11-04-14-27/Ep_636_mae_76.18496276946838_mse_133.15283165001156.pth'
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


