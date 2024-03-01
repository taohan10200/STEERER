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
    loss_weight = [1., 1/2, 1/4., 1/8.],
    sigma = [4],
    gau_kernel_size = 15,
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
    name='NWPU',
    root='../ProcessedData/NWPU/',
    test_set='test.txt',
    train_set='train.txt',
    loc_gt = 'test_gt_loc.txt',
    num_classes=len(network['resolution_num']),
    den_factor=100,
    extra_train_set =None
)



optimizer = dict(
    NAME='adamw',
    BASE_LR=1e-4,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-5,
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
    base_size=None,
    batch_size_per_gpu=8,
    shuffle=True,
    begin_epoch=0,
    end_epoch=800,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path='exp/NWPU/MocHRBackbone_hrnet48/NWPU_final_2023-10-28-16-56/',
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_span =  [-800 -600, -400, -200, -200, -100, -100],
    downsamplerate= 1,
    ignore_label= 255
)

test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=4096,
    loc_base_size=4096,
    loc_threshold = 0.10 ,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,
    model_file=  'exp/NWPU/MocHRBackbone_hrnet48/NWPU_final_2023-10-28-16-56/Ep_617_mae_32.569058299681174_mse_80.39371260587879.pth'#'
    
    
    # model_file = './exp/NWPU/MocHRBackbone_hrnet48/NWPU_HR_base_2022-11-04-01-01/Ep_715_mae_30.73060810663458_mse_72.68940780485204.pth' # Ep_614_mae_31.279445421263574_mse_71.21658028909538.pth'

        #  test 4096: mae: 64.06/311.09
        #  test 2048:mae: 77
        #  mae:  63.6779, mse:  309.7973,
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)

# -----Localization performance with box annotations-----
# AP_small: 0.7266841537920595
# AR_small: 0.6828295989514259
# F1m_small: 0.7040746474908515
# AR_small_category: [0.42393291 0.36564436 0.73972631 0.84076178 0.8427917  0.66378162]
#     avg: 0.6461064473509256
# AP_large: 0.7937366597389477
# AR_large: 0.7458355631044619
# F1m_large: 0.7690409320060393
# AR_large_category: [0.47901598 0.46454416 0.7978246  0.88839592 0.89029675 0.74101198]
#     avg: 0.7101815664678331
# -----Localization performance with points annotations-----
# avg precision_overall:0.0
# avg recall_overall:0.0
# avg F1_overall:0.0
# Mean Loclization Error:11.071264797722778
# -----Counting performance-----
# MAE: 68.13228172825124
# MSE: 339.16208533951306
# NAE: 0.134009762841654
# mae:  68.1323, mse:  339.1621,                nae:  0.1340, Class IoU:
# Mins: 54
# Done