# _base_ =[
#     '../_base_/datasets/imagenet_bs64_swin_224.py'
# ]
gpus = (0, 1,)
log_dir = 'exp'
workers = 6
print_freq = 30
seed = 3035

network = dict(
    backbone="MocHRBackbone",
    sub_arch='hrnet48',
    counter_type = 'withMOE', #'withMOE' 'baseline'
    resolution_num = [0,1,2,3],
    loss_weight = [1., 1./2, 1./4, 1./8],
    sigma = [4],
    gau_kernel_size = 11,
    baseline_loss = False,
    pretrained_backbone="../PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth",


    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=96,
        stages_channel = [384, 192, 96, 48],
        inter_layer=[64,32,16],
        out_channels=1)
    )

dataset = dict(
    name='SHHA',
    root='../ProcessedData/SHHA/',
    test_set='test.txt',
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
    end_epoch=600,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path=None,
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_span = [ -600, -400, -200, -100, -100],
    downsamplerate= 1,
    ignore_label= 255
)


test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=2048,
    loc_base_size=(768,2048),
    loc_threshold=0.2,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,


    # For crowd counting
    # model_file= './exp/SHHA/MocHRBackbone_hrnet48/SHHA_HR_2022-10-25-20-1_251_mae_54.5_mse_86.9/Ep_251_mae_54.571960030021245_mse_86.92610575458059.pth'

    # For localization
    model_file='', # './exp/SHHA/MocHRBackbone_hrnet48/SHHA_HR_2022-10-25-20-1_251_mae_54.5_mse_86.9/final_state.pth'
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


