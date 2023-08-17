
gpus = (0, 1,)
log_dir = 'exp'
workers = 12
print_freq = 30
seed = 3035

network = dict(
    backbone="VGGBackbone",
    sub_arch='vgg19_bn',
    counter_type = 'withMOE', #'withMOE' 'baseline'
    resolution_num = [0,1,2],
    sigma = [4],
    gau_kernel_size = 15,
    baseline_loss = True,
    pretrained_backbone="../PretrainedModels/vgg/",
    # '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/imagenet/CCFBackbone_hrt_base/ccformer_b_2022-08-16-14-59/Ep_142_acc1_75.12800201416016.pth',

    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=96,
        stages_channel = [512, 512, 256],
        inter_layer=[64,32,16],
        out_channels=1)
    )

dataset = dict(
    name='SHHA',
    root='../ProcessedData/SHHA/',
    test_set='test_val.txt',
    train_set='train.txt',
    num_classes=1,
    den_factor=100,
    extra_train_set =None
)

# optimizer = dict(
#     NAME='AdamW',
#     BASE_LR=1e-5,
#     BETAS=(0.9, 0.999),
#     WEIGHT_DECAY=0.00001,
#     AMSGRAD = False
#     )

optimizer = dict(
    NAME='adamw',
    BASE_LR=1e-5,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-4,
    EPS= 1.0e-08,
    MOMENTUM= 0.9,
    AMSGRAD = False,
    NESTEROV= True,
    )

# lr_config = dict(
#     NAME='WarmupMultiStepLR',
#     WARMUP_METHOD='linear',
#     WARMUP_ITERS=5,   # the number of epochs to warmup the lr_rate
#     WARMUP_FACTOR=0.01,
#     STEPS=[100, 200, 300, 400],
#     GAMMA=0.5)

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
    batch_size_per_gpu=8,
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
    scale_factor=16,
    val_span = [-1000, -900, -800,-600, -400, -200, -200],
    downsamplerate= 1,
    ignore_label= 255
)


test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=2048,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,

    # model_file= './exp/SHHA/MocHRBackbone_hrnet48/SHHA_mocHR_small_2022-09-22-20-47_baseline/Ep_319_mae_57.58954155052101_mse_96.65610938266916.pth'
    model_file= './exp/SHHA/MocHRBackbone_hrnet48/SHHA_mocHR_small_2022-09-22-20-39_reference/Ep_441_mae_56.34423482287061_mse_98.67360699428517.pth'
    # model_file= './exp/SHHA/MocHRBackbone_hrnet48/SHHA_mocHR_small_2022-09-23-11-25_refence_512/Ep_559_mae_56.20006972092848_mse_96.45367604706817.pth'
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


