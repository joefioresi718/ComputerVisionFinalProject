# python train_autocast.py --params params_linear --linear --run_id="" 
############ dataloader realated params
import numpy as np
import math
num_workers = 4
batch_size = 24#8
# batch_size = 56#8
# batch_size = 56#8
data_percentage = 1.0#1.0
v_batch_size = 24#80
num_modes = 10#5
cropping_facs = [0.8]

# fix_skip = 2
fix_skip = 4
sr_ratio = 4


############ model input params
# num_frames = 16
dataset = 'ucf101'

num_frames = 8
reso_h = 224 #112
reso_w = 224 #112
# reso_h = 112
# reso_w = 112

ori_reso_h = 240
ori_reso_w = 320

RGB = True
# normalize = False#True
normalize = False#False (default) #True #True


###### Training optimization related params
learning_rate = 1e-2#1e-3 #1e-5
num_epochs = 100
# warmup_array = list(np.linspace(0,1, 10) + 1e-9)
# warmup = len(warmup_array)
lr_scheduler = "patience_based"#"patience_based" #"loss_based (default)" #cosine
cosine_lr_array = list(np.linspace(0.01,1, 5)) + [(math.cos(x) + 1)/2 for x in np.linspace(0,math.pi/0.99, num_epochs-5)]


scheduler_patience = 1
lr_reduce_factor = 2

warmup = True
warmup_array = [0.1, 1]
val_freq = 3
opt_type = 'adam'
# opt_type = 'sgd'
# opt_type = 'adamW'



############## model related params
finetuning_mode = '2d3d' #'3donly' (default), '2donly' '2d3d'
linear = True
backbone = 'R50' #'VIT'
frozen_bb = False
frozen_bn = False
kin_pretrained = False
num_classes = 102
num_encoder_layers = 6
num_att_heads = 8
num_dims = 512
bb_pretrained = 'scratch'

pretrained_checkpoint = None #for scratch-temporal baseline
if pretrained_checkpoint is not None:
    pretrained_checkpoint = pretrained_checkpoint.replace('-symlink', '')
    
aspect_ratio_aug = False
min_crop_factor_training = 0.6
casia_split = 'doesnt_matter'
ft_dropout = 0

val_array = [] #[0,5,10] + list(range(12, num_epochs,2)) #could be empty to take defaults
semi_frozen_bb = False
hemi_frozen_bb = False
eval_only = False
###
load_full_head = True
frozen_3d = False
drop_fc_loading = True