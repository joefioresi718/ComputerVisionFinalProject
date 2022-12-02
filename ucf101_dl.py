import os, sys, traceback
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config as cfg
import random
import json
import cv2
import time
import torchvision.transforms as trans
import parameters as params


# Training dataset.
class ucf101_ar_train_dataset(Dataset):

    def __init__(self, params, shuffle=True, data_percentage=1.0, split=1):
        self.params = params
        
        if split <= 3:
            self.all_paths = open(os.path.join(cfg.ucf101_path, f'ucfTrainTestlist/trainlist0{split}.txt'),'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')
        self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
        # Label building.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
        label = self.classes[vid_path.split('/')[-2]]  # This element should be activity name.

        # Clip building.
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, idx
    
    def build_clip(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            self.ori_reso_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.ori_reso_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)
            ############################# frame_list maker start here #################################

            skip_frames_full = self.params.fix_skip #frame_count/(self.params.num_frames)

            left_over = frame_count - self.params.fix_skip*self.params.num_frames

            if left_over > 0:
                start_frame_full = np.random.randint(0, int(left_over)) 
            else:
                skip_frames_full /= 2
                left_over = frame_count - skip_frames_full*self.params.num_frames
                start_frame_full = np.random.randint(0, int(left_over)) 

            frames_full = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            # Some edge case fixing.
            if frames_full[-1] >= frame_count:
                # print('some corner case not covered')
                frames_full[-1] = int(frame_count-1)
            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################
            full_clip = []
            list_full = []
            count = -1
            random_array = np.random.rand(2,10)
            x_erase = np.random.randint(0,self.params.reso_w, size = (2,))
            y_erase = np.random.randint(0,self.params.reso_h, size = (2,))
            # On an average cropping, factor is 80% i.e. covers 64% area.
            cropping_factor1 = np.random.uniform(self.params.min_crop_factor_training, 1, size = (2,)) 
            x0 = np.random.randint(0, (self.ori_reso_w - self.ori_reso_w*cropping_factor1[0]) + 1)
            y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[0]) + 1)

            # Here augmentations are not strong as self-supervised training.
            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
            gamma1 = np.random.uniform(0.85,1.15, size = (2,))
            erase_size1 = np.random.randint(int((self.ori_reso_h/6)*(self.params.reso_h/224)),int((self.ori_reso_h/3)*(self.params.reso_h/224)), size = (2,))
            erase_size2 = np.random.randint(int((self.ori_reso_w/6)*(self.params.reso_h/224)),int((self.ori_reso_w/3)*(self.params.reso_h/224)), size = (2,))
            random_color_dropped = np.random.randint(0,3,(2))

            while(cap.isOpened()): 
                count += 1
                ret, frame = cap.read()
                if ((count not in frames_full)) and (ret == True): 
                    continue
                if ret == True:
                    if (count in frames_full):
                        full_clip.append(self.augmentation(frame, random_array[0], x_erase, y_erase, cropping_factor1[0],\
                            x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                            gamma1[0],erase_size1,erase_size2, random_color_dropped[0]))
                        list_full.append(count)
                else:
                    break

            if len(full_clip) < self.params.num_frames and len(full_clip)>(self.params.num_frames/2) :
                print(f'Clip {vid_path} is missing {self.params.num_frames-len(full_clip)} frames.')
                remaining_num_frames = self.params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
                list_full = list_full + list_full[::-1][1:remaining_num_frames+1]

            try:
                assert(len(full_clip) == self.params.num_frames)
                return full_clip, list_full
            except:
                print(frames_full)
                print(f'Clip {vid_path} Failed')
                return None, None   
        except:
            # traceback.print_exc()
            print(f'Clip {vid_path} Failed')
            return None, None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(image)
  
        image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w))

        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) 
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) 
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) 
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1)
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[7] < 0.4 :
            image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        if random_array[8] <0.4 :
            image = trans.functional.erase(image, x_erase[1], y_erase[1], erase_size1[1], erase_size2[1], v=0) 

        return image


    
# Validation dataset.
class ucf101_ar_val_dataset(Dataset):

    def __init__(self, params, shuffle=True, data_percentage=1.0, mode=0, hflip=0, cropping_factor=0.8, split=1):
        
        self.params = params
        self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']

        if split <= 3:
            self.all_paths = open(os.path.join(cfg.ucf101_path, f'ucfTrainTestlist/testlist0{split}.txt'),'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')    
                
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.hflip = hflip
        self.cropping_factor = cropping_factor
        if self.cropping_factor == 1:
            self.output_reso_h = int(params.reso_h/0.8)
            self.output_reso_w = int(params.reso_w/0.8)
        else:
            self.output_reso_h = int(params.reso_h)
            self.output_reso_w = int(params.reso_w)                       
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
        # Label building.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
        label = self.classes[vid_path.split('/')[-2]]  # This element should be activity name.

        # Clip building.
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, idx

    def build_clip(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)
            self.ori_reso_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.ori_reso_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)

            N = frame_count
            n = self.params.num_frames

            skip_frames_full = self.params.fix_skip 

            if skip_frames_full*n > N:
                skip_frames_full /= 2

            left_over = skip_frames_full*n
            F = N - left_over

            start_frame_full = 0 + int(np.linspace(0, F-10, params.num_modes)[self.mode])

            if start_frame_full < 0:
                start_frame_full = self.mode

            full_clip_frames = []

            full_clip_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(self.params.num_frames)])

            count = -1
            full_clip = []
            list_full = []

            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read()
                if ((count not in full_clip_frames) and (ret == True)):
                    continue
                if ret == True:
                    if (count in full_clip_frames):
                        full_clip.append(self.augmentation(frame))
                        list_full.append(count)
                else:
                    break
            # Appending the remaining frames in case of clip length < required frames.
            if len(full_clip) < self.params.num_frames and len(full_clip)>(self.params.num_frames/2):
                remaining_num_frames = self.params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
                list_full = list_full + list_full[::-1][1:remaining_num_frames+1]
            assert (len(full_clip) == self.params.num_frames)

            return full_clip, list_full
        except:
            traceback.print_exc()
            print(f'Clip {vid_path} Failed, frame_count {frame_count}.')
            return None, None

    def augmentation(self, image):
        image = self.PIL(image)

        if self.cropping_factor <= 1:
            image = trans.functional.center_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor)))

        image = trans.functional.resize(image, (self.output_reso_h, self.output_reso_w))

        if self.hflip !=0:
            image = trans.functional.hflip(image)

        return trans.functional.to_tensor(image)


def collate_fn(batch):
    f_clip, label, vid_path, idx_list = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) 
            label.append(item[1])
            vid_path.append(item[2])
            idx_list.append(torch.tensor(item[3]))

    f_clip = torch.stack(f_clip, dim=0)
    
    return f_clip, label, vid_path, idx_list 


if __name__ == '__main__':
    import torchvision
    from PIL import Image, ImageDraw, ImageFont
    import parameters as params

    visualize = True
    run_id = 'ucf101'
    
    vis_output_path = os.path.join('some_visualization', run_id)
    
    train_dataset = ucf101_ar_train_dataset(params=params, shuffle=False, data_percentage=params.data_percentage)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers)

    print(f'Length of training dataset: {len(train_dataset)}')
    print(f'Steps involved: {len(train_dataset)/params.batch_size}')
    t = time.time()

    for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
        if i % 10 == 0:
            print()
            clip = clip.permute(0, 1, 3, 4, 2)
            if params.RGB or params.normalize:
                clip = torch.flip(clip, [4])
            if visualize:
                classes = json.load(open(cfg.ucf101_class_mapping))['classes']
                inv_map = {v: k for k, v in classes.items()}
                if not os.path.exists(vis_output_path):
                    os.makedirs(vis_output_path)
                counter = 0
                for kk in range(clip.shape[0]):
                    clip1 = clip[kk]
                    for kk1, frame in enumerate(clip[kk]):
                        msg = ''
                        frame = trans.functional.to_pil_image(frame.permute(2,0,1))
                        d1 = ImageDraw.Draw(frame)
                        myFont = ImageFont.load_default()
                        # msg = str(frame_list[kk][kk1].item())
                        msg = str(kk)
                        d1.text((100, 0), msg, font=myFont, fill=(255, 255, 255))
                        
                        if kk1 == 0:
                            msg1 = inv_map[label[kk]]
                            d1.text((50, 125), msg1, font=myFont, fill=(255, 255, 255))
                        
                        frame = trans.functional.to_tensor(frame)
                        print(frame.shape)
                        clip1[kk1] = frame.permute(1,2,0)
                    clip1 *= 255
                    clip1 = clip1.to(torch.uint8).permute(0,3,1,2)      
                    image = torchvision.utils.make_grid(clip1, nrow=params.num_frames)
                    filename = os.path.join(vis_output_path, f'{counter}.png') 
                    torchvision.io.write_png(image, filename)
                    counter += 1
            break
            
    print(f'Time taken to load data is {time.time()-t}')
      

    validation_dataset = ucf101_ar_val_dataset(params=params, split=1, shuffle=False, data_percentage=params.data_percentage, mode=2)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers)

    print(f'Length of validation dataset: {len(validation_dataset)}')
    print(f'Steps involved: {len(validation_dataset)/params.v_batch_size}')
    t = time.time()

    for i, (clip, label, vid_path, frame_list) in enumerate(validation_dataloader):
        if i % 25 == 0:
            print()
            print(f'Full_clip shape is {clip.shape}')
            print(f'Label is {label}')

    print(f'Time taken to load data is {time.time()-t}')
