# _base_ =[
#     '../_base_/datasets/imagenet_bs64_swin_224.py'
# ]
gpus = (0, 1,)
log_dir = 'exp'
workers = 12
print_freq = 30
seed = 3035
network = dict(
    backbone="CCFBackbone",
    sub_arch='hrt_base',
    pretrained_backbone="/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/imagenet/CCFBackbone_hrt_base/ccformer_b_2022-08-24-00-27/model_best.pth.tar",
    # '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/imagenet/CCFBackbone_hrt_base/ccformer_b_2022-08-16-14-59/Ep_142_acc1_75.12800201416016.pth',
    head=dict(
        type='CountingHead',
        in_channels=78 + 156 + 312 + 624,
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
    DECAY_EPOCHS=150,
    DECAY_RATE = 0.1,
    WARMUP_EPOCHS=50,   # the number of epochs to warmup the lr_rate
    WARMUP_LR=5.0e-07,
    MIN_LR= 1.0e-06
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
    route_size=(32, 32),  # height, width
    base_size=2048,
    batch_size_per_gpu=4,
    shuffle=True,
    begin_epoch=0,
    end_epoch=600,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path=None, #'./exp/NWPU/seg_hrnet/hrt_small_2022-08-07-23-15/',
    # './exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-05-27-15-03'
    flip=True,
    multi_scale=True,
    scale_factor=16,
    val_span = [-600, -300, -200, -100, -50],
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
    model_file= '', #'./exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-06-03-23-12/Ep_280_mae_54.884169212251905_mse_226.06904272422108.pth'
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)