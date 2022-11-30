import numpy as np


# Dataset parameters.
num_classes = 102
num_frames = 16
fix_skip = 2
num_modes = 5
num_skips = 1
data_percentage_ucf101 = 0.01

######################
# Training parameters.
batch_size_ucf101 = 4
v_batch_size = 4
learning_rate = 1e-4
######################
num_workers = 0
num_epochs = 100
warmup_array = list(np.linspace(0.01, 1, 5) + 1e-9)
warmup = len(warmup_array)
scheduled_drop = 2
lr_patience = 0

# Transformer parameters.
patch_size = 16

# Validation augmentation params.
hflip = [0]
cropping_fac1 = [0.8]
casia_split = 'nothing'
RGB = True
# normalize = False#True
normalize = False#False (default) #True #True

# Training augmentation params.
reso_h = 112
reso_w = 112
ori_reso_h = 240
ori_reso_w = 320

# Tracking params.
wandb = False
