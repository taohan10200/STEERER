# _base_ =[
#     '../_base_/datasets/imagenet_bs64_swin_224.py'
# ]
gpus = (0, 1,)
log_dir = 'exp'
workers = 12
print_freq = 30
seed = 3035

network = dict(
    backbone="MAEvitBackbone",
    sub_arch='vit_base',
    counter_type = 'single_resolution', #'withMOE' 'single_resolution'

    resolution_num = [0,1,2,3],
    sigma = [4],
    gau_kernel_size = 15,
    baseline_loss = False,
    pretrained_backbone= '../PretrainedModels/vitdet/vit_base_pretrain/vitbase_lr1e3_stepLRx3_bmp1_adafactor_wd01_layerdecay075_lpe_peddet_citypersons_size1024_LSA_reduct8_tbn1_heads2_gate1_expandBS_expr2_setting_SampleWeight3_SharePosEmbed.pth',
    #"../PretrainedModels/vitdet/ckpt_task0_iter_newest.pth.tar",

    # "../PretrainedModels/maevit/mae_pretrain_vit_base.pth",
    # '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/imagenet/CCFBackbone_hrt_base/ccformer_b_2022-08-16-14-59/Ep_142_acc1_75.12800201416016.pth',

    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=768,
        stages_channel = [768],
        inter_layer=[64,32,16],
        out_channels=1)
    )

dataset = dict(
    name='QNRF',
    root='../ProcessedData/QNRF/',
    test_set='test_val.txt',
    train_set='train.txt',
    num_classes=1,
    den_factor=100,
    extra_train_set =None
)

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
    image_size=(768, 1024),  # height width
    route_size=(256, 256),  # height, width
    base_size=2048,
    batch_size_per_gpu=4,
    shuffle=True,
    begin_epoch=0,
    end_epoch=1000,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path=None,#"./exp/QNRF/MocCatBackbone_hrnet48/QNRF_HRCat_2022-09-28-23-32",#'./exp/QNRF/MocHRBackbone_hrnet48/QNRF_mocHR_small_2022-09-21-01-23/', #'/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/SHHB/MocHRBackbone_hrnet48/mocHR_small_2022-09-18-14-54/', #'/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-13-16-28/', # '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-09-21-07/', #'./exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-06-20-35/', #'/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-06-13-03/', #'./exp/NWPU/MocHRBackbone_hrnet48/mocHR_small_2022-09-04-19-21/', #'./exp/NWPU/MocBackbone_moc_small/moc_small_2022-08-31-14-41/', #'./exp/NWPU/seg_hrnet/hrt_small_2022-08-07-23-15/',
    # './exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-05-27-15-03'
    flip=True,
    multi_scale=True,
    scale_factor=16,
    val_span = [-1000, -600, -600,-400, -400, -200, -200],
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
    # model_file= './exp/QNRF/MocHRBackbone_hrnet48/QNRF_mocHR_small_2022-09-21-01-23/Ep_287_mae_80.81543667730458_mse_134.10439891983856.pth', #'./exp/NWPU/seg_hrnet/seg_hrnet_w48_2022-06-03-23-12/Ep_280_mae_54.884169212251905_mse_226.06904272422108.pth'
    model_file = './exp/QNRF/MocHRBackbone_hrnet48/QNRF_mocHR_small_2022-09-23-20-07/Ep_617_mae_78.82519394195009_mse_136.08208312773016.pth'
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


