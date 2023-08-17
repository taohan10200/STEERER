# _base_ =[
#     '../_base_/datasets/imagenet_bs64_swin_224.py'
# ]
gpus = (0, 1,)
log_dir = 'exp'
workers = 12
print_freq = 30
seed = 3035
network = dict(
    backbone="HRTBackbone",
    sub_arch='hrt_small',
    pretrained_backbone='/mnt/petrelfs/hantao/PretrainedModels/hrt/hrt_small.pth',
    head=dict(
        type='CountingHead',
        in_channels=32 + 64 + 128 + 256,
        inter_layer=[
            256,
            128,
            64],
        out_channels=1))
dataset = dict(
    name='NWPU',
    root='../ProcessedData/NWPU_2048/',
    test_set='test_val.txt',
    train_set='train.txt',
    num_classes=1,
    den_factor=200,
    extra_train_set =None
)

optimizer = dict(
    NAME='AdamW',
    BASE_LR=1e-5,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=0.00000001,
    AMSGRAD = False
    )

lr_config = dict(
    NAME='WarmupMultiStepLR',
    WARMUP_METHOD='linear',
    WARMUP_ITERS=5,   # the number of epochs to warmup the lr_rate
    WARMUP_FACTOR=0.01,
    STEPS=[100, 200, 300, 400],
    GAMMA=0.5)


total_epochs = 210

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

train = dict(
    counter='normal',
    image_size=(1024, 1024),  # height width
    route_size=(32, 32),  # height, width
    base_size=2048,
    batch_size_per_gpu=4,
    shuffle=True,
    begin_epoch=0,
    end_epoch=1000,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path='./exp/NWPU/hrt_small/hrformer_s_2022-08-11-00-04/', #'./exp/NWPU/seg_hrnet/hrt_small_2022-08-07-23-15/',
    # './exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-05-27-15-03'
    flip=True,
    multi_scale=True,
    scale_factor=16,
    val_span = [-1000, -1000, -500, -100, -50],
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
    # './exp/NWPU/seg_hrnet/seg_hrnet_w48_nwpu_2022-06-03-23-12/Ep_138_mae_45.79466183813661_mse_116.98580130706075.pth'
    model_file='./exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-06-03-23-12/Ep_280_mae_54.884169212251905_mse_226.06904272422108.pth'
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)