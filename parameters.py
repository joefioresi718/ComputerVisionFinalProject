import numpy as np


# Dataset parameters.
num_classes = 102
num_frames = 16
fix_skip = 2
num_modes = 5
num_skips = 1
data_percentage = 1.0
transformer_size = 'small'

######################
# Training parameters.
batch_size = 16
v_batch_size = 16
learning_rate = 1e-4
######################
num_workers = 4
num_epochs = 100
warmup_array = list(np.linspace(0.01, 1, 5) + 1e-9)
warmup = len(warmup_array)
scheduled_drop = 2
lr_patience = 0

# Transformer parameters.
patch_size = 16

# Validation augmentation params.
hflip = [0]
cropping_facs = [0.8]
RGB = True
normalize = False

# Training augmentation params.
reso_h = 224
reso_w = 224
ori_reso_h = 240
ori_reso_w = 320
min_crop_factor_training = 0.6

# Tracking params.
wandb = False
