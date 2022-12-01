import glob
import os.path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
import random
import time
import cv2

import config as cfg
import parameters as params


# UCF 101 action recognition dataset.
class ucf101_ar_dataset(Dataset):
    def __init__(self, data_split, cv_split=1, shuffle=True, data_percentage=1.0, mode=0):
        self.data_split = data_split
        # Use config.py pathing to find data splits.
        data_list = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'{data_split}list{cv_split:02d}.txt'),'r').read().splitlines()
        data_list = [x.split(' ')[0] for x in data_list]

        # Use config.py pathing to find class indexes.
        class_list = open(os.path.join(cfg.class_mapping_ucf101)).read().splitlines()
        self.class_list = {cls_num.split(' ')[1]: int(cls_num.split(' ')[0]) for cls_num in class_list}

        if shuffle:
            random.shuffle(data_list)

        # Data limiter.
        self.data_percentage = data_percentage
        data_limit = int(len(data_list)*self.data_percentage)
        self.data = data_list[0: data_limit]

        # Augmentation parameters.
        self.mode = mode
        self.erase_size = 19

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Finds clip using config.py pathing.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split('/')[0], os.path.basename(self.data[idx].strip()))
        video_frames, frame_pos = self.read_video(vid_path)
        label = self.class_list[self.data[idx].split('/')[0]]
        return video_frames, int(label) - 1, vid_path, idx

    # Frame augmentation function.
    def augmentation(self, image):
        if self.data_split == 'train':
            # Compute augmentation strengths.
            x_erase = np.random.randint(0, params.reso_h, size=(2,))
            y_erase = np.random.randint(0, params.reso_w, size=(2,))
            # An average cropping factor is 80% i.e. covers 64% area.
            cropping_factor1 = np.random.uniform(0.6, 1, size=(2,))
            x0 = np.random.randint(0, params.ori_reso_w - params.ori_reso_w*cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, params.ori_reso_h - params.ori_reso_h*cropping_factor1[0] + 1)
            contrast_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            hue_factor1 = np.random.uniform(-0.05, 0.05, size=(2,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            brightness_factor1 = np.random.uniform(0.9, 1.1,size=(2,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
            erase_size1 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            erase_size2 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            random_color_dropped = np.random.randint(0, 3, (2))

            # Convert to PIL for transforms. 
            image = trans.functional.to_pil_image(image)

            # Always resize crop the frame.
            image = trans.functional.resized_crop(image, y0, x0, int(params.ori_reso_h*cropping_factor1[0]), 
                                                  int(params.ori_reso_h*cropping_factor1[0]), (params.reso_h, params.reso_w))

            # Random augmentation probabilities.
            random_array = np.random.rand(8)

            if random_array[0] < 0.125/2:
                image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1[0]) # 0.75 to 1.25
            if random_array[1] < 0.3/2 :
                image = trans.functional.adjust_hue(image, hue_factor=hue_factor1[0]) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
            if random_array[2] < 0.3/2 :
                image = trans.functional.adjust_saturation(image, saturation_factor=saturation_factor1[0]) # brightness factor will be between [0.75, 1,25]
            if random_array[3] < 0.3/2 :
                image = trans.functional.adjust_brightness(image, brightness_factor=brightness_factor1[0]) # brightness factor will be between [0.75, 1,25]
            if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
                image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1[0]) # 0.75 to 1.25
            if random_array[4] > 0.9:
                image = trans.functional.rgb_to_grayscale(image, num_output_channels=3)
                if random_array[5] > 0.25:
                    image = trans.functional.adjust_gamma(image, gamma=gamma1[0], gain=1) # gamma range [0.8, 1.2]
            if random_array[6] > 0.5:
                image = trans.functional.hflip(image)

            # Convert frame to tensor.
            image = trans.functional.to_tensor(image)

            if random_array[6] < 0.5/2 :
                image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        else:
            cropping_factor1 = np.random.uniform(0.6, 1, size=(2,))
            image = trans.functional.to_pil_image(image)
            if cropping_factor1[0] <= 1:
                image = trans.functional.center_crop(image, (int(params.ori_reso_h*cropping_factor1[0]), int(params.ori_reso_h*cropping_factor1[0])))
            image = trans.functional.resize(image, (params.reso_h, params.reso_w))
            if params.hflip[0] != 0:
                image = trans.functional.hflip(image)
            # Convert frame to tensor.
            image = trans.functional.to_tensor(image)
        return image

    # Clip builder, reads video frames from custom start and end times (if necessary), stacks them.
    def read_video(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            total_frames = cap.get(7)
            F = total_frames - params.fix_skip*params.num_frames
            start = int(np.linspace(0, F-10, params.num_modes)[self.mode])
            if start < 0:
                start = self.mode
            full_clip_frames = start + np.asarray([int(int(params.fix_skip)*f) for f in range(params.num_frames)])

            video_frames = []
            frame_pos = []
            count = -1
            # Loops from start to end, adding frames and timestamps to list.
            while cap.isOpened():
                count += 1
                ret, frame = cap.read()
                if count not in full_clip_frames and ret == True:
                    continue
                else:
                    if ret == True:
                        video_frames.append(self.augmentation(frame))
                        frame_pos.append(count)
                    else:
                        break

            # In case of size mismatch.
            if len(video_frames) < params.num_frames and len(video_frames)>(params.num_frames/2):
                remaining_num_frames = params.num_frames - len(video_frames)
                video_frames = video_frames + video_frames[::-1][1:remaining_num_frames+1]

            # Ensure correct clip shape.
            assert (len(video_frames) == params.num_frames)
            # Stack it into a tensor.
            video = torch.stack(video_frames, 0)

            return video, frame_pos
        except:
            return None, None


if __name__ == '__main__':
    train_dataset = ucf101_ar_dataset(data_split='test', shuffle=False, mode=0, data_percentage=0.2)
    
    # Demonstrate loading steps and time.
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size_ucf101, shuffle=True, 
                                  num_workers=params.num_workers)
    print(f'Steps involved: {len(train_dataset)/params.batch_size_ucf101}')
    print(f'Length: {len(train_dataset)}')
    t = time.time()

    # for i, (clip, label, paths, frame_pos) in enumerate(train_dataloader):
    #     if i % 10 == 0:
    #         print()
    #         clip = clip.permute(0, 1, 3, 4, 2)
    #         print(f'Full_clip shape is {clip.shape}')
    #         print(f'Label is {label}')

    print(f'Time taken to load data is {time.time()-t}')

    # Sample visualization.
    clip, label, paths, frame_pos = train_dataset[15]
    # clip = torch.stack(clip, dim=0)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'DL Clip Visualization Label: {label}')
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(clip[i, ...].permute(1, 2, 0))
        plt.title(f'Vid Idx {frame_pos}', y=-.18)
        plt.axis('off')
    plt.show()
