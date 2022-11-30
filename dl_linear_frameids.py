'''
Time to load 133.0
'''
import os, sys, traceback
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# sys.path.insert(0, "/sensei-fs/users/idave/ucf101_exp_with_VTN/")

import config as cfg
import random
import pickle
# import parameters_BL as params
import json
import math
import cv2
# from tqdm import tqdm
import time
import torchvision.transforms as trans
# from decord import VideoReader

class baseline_dataloader_train_strong(Dataset):

    def __init__(self, params, dataset='ucf101', shuffle = True, data_percentage = 1.0, split = 1, frame_wise_aug = False, no_aug= False):
        # supported datasets: ucf101, hmdb51
        
        self.dataset= dataset
        
        self.params = params
        
        if self.dataset == 'ucf101':
            
            if split == 1:
                self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist01.txt'),'r').read().splitlines()
            elif split ==2: 
                self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist02.txt'),'r').read().splitlines()
            elif split ==3: 
                self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist03.txt'),'r').read().splitlines()
            else:
                print(f'Invalid split input: {split}')
            self.classes= json.load(open(cfg.class_mapping))['classes']

        elif self.dataset == 'hmdb51':
            file_name = 'hmdb_train_' + str(split) + '.txt'
            self.all_paths = open(os.path.join(cfg.path_folder,file_name),'r').read().splitlines()
            self.classes= json.load(open(cfg.hmdb_mapping))
        elif self.dataset == 'k400':
            self.all_paths = open('/sensei-fs/users/idave/data/k400train_full_path_resized_annos.txt','r').read().splitlines()
        else:
            print(f'{self.dataset} dne')
            
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19
        self.framewise_aug = frame_wise_aug
        self.no_aug = no_aug

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
    
        # label_building
        if self.dataset == 'ucf101':
            vid_path = cfg.path_folder + '/UCF-101/' + self.data[idx].split(' ')[0]
            label = self.classes[vid_path.split('/')[-2]] # This element should be activity name
        elif self.dataset == 'hmdb51':
            vid_path = self.data[idx]
            label = self.classes[vid_path.split(' ')[1]]
            vid_path = cfg.path_folder  + '/hmdb/' + vid_path.split(' ')[1]+ '/' + vid_path.split(' ')[0]
        elif self.dataset == 'k400':
            vid_path = cfg.path_folder + '/Kinetics/kinetics-dataset/' + self.data[idx].split(' ')[0]
            # label = -1 # I need to put label info in the full paths yet
            label = int(self.data[idx].split(' ')[1]) - 1 
            
        # clip_building
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, frame_list
    
    def build_clip(self, vid_path):
        

        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            self.ori_reso_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.ori_reso_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)
            ############################# frame_list maker start here#################################

            skip_frames_full = self.params.fix_skip #frame_count/(self.params.num_frames)

            left_over = frame_count - self.params.fix_skip*self.params.num_frames

            if left_over>0:
                start_frame_full = np.random.randint(0,int(left_over)) 
            else:
                skip_frames_full /= 2
                left_over = frame_count - skip_frames_full*self.params.num_frames
                start_frame_full = np.random.randint(0,int(left_over)) 
                # print(f'starting frame is set 0 for {vid_path}')
            frames_full = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])



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


            cropping_factor1 = np.random.uniform(self.params.min_crop_factor_training, 1, size = (2,)) # on an average cropping factor is 80% i.e. covers 64% area
            
            if not self.params.no_ar_distortion:
                x0 = np.random.randint(0, (self.ori_reso_w - self.ori_reso_w*cropping_factor1[0]) + 1)
                if self.params.aspect_ratio_aug:
                    y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[1]) + 1)
                else:
                    # y0 = np.random.randint(0, (self.params.ori_reso_h - self.ori_reso_h*cropping_factor1[0]) + 1)
                    #######the above one was still incorrect for k400!#########
                    y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[0]) + 1)

            else:
                x0 = np.random.randint(0, (self.ori_reso_w - self.min_size*cropping_factor1[0]) + 1)
                y0 = np.random.randint(0, (self.ori_reso_h - self.min_size*cropping_factor1[0]) + 1)


            #Here augmentations are not strong as self-supervised training
            # if not self.framewise_aug:
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
                    if self.framewise_aug:
                        contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
                        hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
                        saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
                        brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
                        gamma1 = np.random.uniform(0.85,1.15, size = (2,))
                        erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
                        erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
                        random_color_dropped = np.random.randint(0,3,(2))

                    if (count in frames_full):
                        if self.params.weak_aug:
                            full_clip.append(self.weak_augmentation(frame, random_array[0], x_erase, y_erase, cropping_factor1[0],\
                                x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                                gamma1[0],erase_size1,erase_size2, random_color_dropped[0]))
                        else:
                            full_clip.append(self.augmentation(frame, random_array[0], x_erase, y_erase, cropping_factor1[0],\
                                x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                                gamma1[0],erase_size1,erase_size2, random_color_dropped[0]))
                        list_full.append(count)
                else:
                    break

            if len(full_clip) < self.params.num_frames and len(full_clip)>(self.params.num_frames/2) :
                print(f'Clip {vid_path} is missing {self.params.num_frames-len(full_clip)} frames')
                remaining_num_frames = self.params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
                list_full = list_full + list_full[::-1][1:remaining_num_frames+1]

            try:
                assert(len(full_clip)==self.params.num_frames)

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
        # image = trans.functional.resized_crop(image,y0,x0,int(self.params.ori_reso_h*cropping_factor1),int(self.params.ori_reso_h*cropping_factor1),(self.params.reso_h,self.params.reso_w))
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w))
        else:
            # image = trans.functional.resized_crop(image,y0,x0,int(self.params.ori_reso_h*cropping_factor1),int(self.params.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w))
            ####### THE ABOVE LINE SEEMS TO HAVE ISSUE WHICH WAS ORIGINALLY USED EVERYWHERE, SHOULD BE FINE FOR UCF BUT NOT FOR OTHERS
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
    
    def weak_augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(image)
        # image = trans.functional.resized_crop(image,y0,x0,int(self.params.ori_reso_h*cropping_factor1),int(self.params.ori_reso_h*cropping_factor1),(self.params.reso_h,self.params.reso_w))
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w))
        else:
            # image = trans.functional.resized_crop(image,y0,x0,int(self.params.ori_reso_h*cropping_factor1),int(self.params.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w))
            ####### THE ABOVE LINE SEEMS TO HAVE ISSUE WHICH WAS ORIGINALLY USED EVERYWHERE, SHOULD BE FINE FOR UCF BUT NOT FOR OTHERS
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w))


        '''if random_array[0] < 0.125/2:
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
        '''
        
        image = trans.functional.to_tensor(image)

        # if random_array[7] < 0.4 :
        #     image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        # if random_array[8] <0.4 :
        #     image = trans.functional.erase(image, x_erase[1], y_erase[1], erase_size1[1], erase_size2[1], v=0) 

        return image

    ####################################################
    
class multi_baseline_dataloader_val_strong(Dataset):

    def __init__(self, params, dataset= 'ucf101', shuffle = True, data_percentage = 1.0, mode = 0, \
                hflip=0, cropping_factor=0.8, split = 1, retrieval_train = False, total_num_modes = 10, casia_split = 'g', threeCrop = False):
        #casia split could be g, p1, p2, p3 
        
        self.total_num_modes = total_num_modes
        self.casia_split = casia_split
        self.params = params
        self.dataset = dataset
        self.threecrop = threeCrop
        if self.dataset=='ucf101':
            self.classes= json.load(open(cfg.class_mapping))['classes']

            if retrieval_train:
                if split == 1:
                    self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist01.txt'),'r').read().splitlines()
                elif split ==2: 
                    self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist02.txt'),'r').read().splitlines()
                elif split ==3: 
                    self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/trainlist03.txt'),'r').read().splitlines()
                else:
                    print(f'Invalid split input: {split}')

            else:

                if split == 1:
                    self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/testlist01.txt'),'r').read().splitlines()
                elif split ==2: 
                    self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/testlist02.txt'),'r').read().splitlines()
                elif split ==3: 
                    self.all_paths = open(os.path.join(cfg.path_folder, 'ucfTrainTestlist/testlist03.txt'),'r').read().splitlines()
                else:
                    print(f'Invalid split input: {split}')        
        
        elif self.dataset=='hmdb51':
            self.classes= json.load(open(cfg.hmdb_mapping))

            if retrieval_train:
                file_name = 'hmdb_train_' + str(split) + '.txt'
                self.all_paths = open(os.path.join(cfg.path_folder,file_name),'r').read().splitlines()

            else:
                # print('here')
                file_name = 'hmdb_test_' + str(split) + '.txt'
                self.all_paths = open(os.path.join(cfg.path_folder,file_name),'r').read().splitlines()
                
        elif self.dataset=='k400':
            if retrieval_train:
                self.all_paths = open('/sensei-fs/users/idave/data/k400train_full_path_resized_annos.txt','r').read().splitlines()


            else:
                # print('here')
                self.all_paths = open('/sensei-fs/users/idave/data/k400val_full_path_resized_annos.txt','r').read().splitlines()
                
                
        elif self.dataset=='casia':
            
            if self.casia_split == 'g':
                self.all_paths = open('/sensei-fs/users/idave/data/casia-b/gallery_vids.txt','r').read().splitlines()
            elif self.casia_split == 'nm':
                self.all_paths = open('/sensei-fs/users/idave/data/casia-b/prob_vids_nm.txt','r').read().splitlines()
            elif self.casia_split == 'cl':
                self.all_paths = open('/sensei-fs/users/idave/data/casia-b/prob_vids_cl.txt','r').read().splitlines()
            elif self.casia_split == 'bg':
                self.all_paths = open('/sensei-fs/users/idave/data/casia-b/prob_vids_bg.txt','r').read().splitlines()

                
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
    
        # label_building
        # label_building
        if self.dataset == 'ucf101':
            vid_path1 = cfg.path_folder + '/UCF-101/' + self.data[idx].split(' ')[0]
            label = self.classes[vid_path1.split('/')[-2]] # This element should be activity name
        elif self.dataset == 'hmdb51':
            vid_path = self.data[idx]
            label = self.classes[vid_path.split(' ')[1]]
            vid_path1 = cfg.path_folder  + '/hmdb/' + vid_path.split(' ')[1]+ '/' + vid_path.split(' ')[0]
            
        elif self.dataset == 'k400':
            vid_path1 = cfg.path_folder + '/Kinetics/kinetics-dataset/' + self.data[idx].split(' ')[0]
            # label = -1 # I need to put label info in the full paths yet
            label = int(self.data[idx].split(' ')[1]) - 1 
            
        elif self.dataset == 'casia':
            vid_path = self.data[idx]
            vid_path1 = vid_path.split(' ')[0]
            label = int(vid_path.split(' ')[1])


        # clip_building
        clip, frame_list = self.build_clip(vid_path1)

        return clip, label, vid_path1, frame_list

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


            start_frame_full = 0 + int(np.linspace(0,F-10, self.total_num_modes)[self.mode])


            if start_frame_full< 0:
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
                        if self.threecrop:
                            full_clip.extend(self.augmentation(frame))
                        else:
                            full_clip.append(self.augmentation(frame))
                        list_full.append(count)

                else:
                    break
            # Appending the remaining frames in case of clip length < required frames
            if not self.threecrop:
                if len(full_clip) < self.params.num_frames and len(full_clip)>(self.params.num_frames/2):
                    remaining_num_frames = self.params.num_frames - len(full_clip)
                    full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
                    list_full = list_full + list_full[::-1][1:remaining_num_frames+1]
            else:
                if len(full_clip) < 3*self.params.num_frames and len(full_clip)>(3*self.params.num_frames/2):
                    remaining_num_frames = 3*self.params.num_frames - len(full_clip)
                    full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
                    list_full = list_full + list_full[::-1][1:remaining_num_frames+1]
            if not self.threecrop:
                assert (len(full_clip) == self.params.num_frames)
            else:
                assert (len(full_clip) == 3*self.params.num_frames)


            return full_clip, list_full

        except:
            # traceback.print_exc()
            print(f'Clip {vid_path} Failed, frame_count {frame_count}')
            return None, None

    def augmentation(self, image):
        image = self.PIL(image)
        

        if self.cropping_factor <= 1:
            # image = trans.functional.center_crop(image,(int(self.params.ori_reso_h*self.cropping_factor),int(self.params.ori_reso_h*self.cropping_factor)))
            
            # image = trans.functional.center_crop(image,(int(self.params.ori_reso_h*self.cropping_factor),int(self.params.ori_reso_h*self.cropping_factor)))
            if self.params.no_ar_distortion:
                image = trans.functional.center_crop(image,(int(self.min_size*self.cropping_factor),int(self.min_size*self.cropping_factor)))
            else:
                image = trans.functional.center_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor)))

                
            if self.threecrop:
                image1 = trans.functional.five_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor))) #torchvision doc says this is non deteministic function, may not always return 5 crops, since I am using bigger overlapping crops, should be fine to just take 2 of the corner crops, let's see how it works. 
                image1_1 = image1[0]
                image1_2 = image1[-2]
                
            
            
            # print(image1.shape) 
            
            
        image = trans.functional.resize(image, (self.output_reso_h, self.output_reso_w))
        if self.threecrop:
            image1_1 = trans.functional.resize(image1_1, (self.output_reso_h, self.output_reso_w))
            image1_2 = trans.functional.resize(image1_2, (self.output_reso_h, self.output_reso_w))
        
        if self.hflip !=0:
            image = trans.functional.hflip(image)
        if self.threecrop:
            return trans.functional.to_tensor(image), trans.functional.to_tensor(image1_1), trans.functional.to_tensor(image1_2)
        
        return trans.functional.to_tensor(image)


def collate_fn1(batch):
    clip, label, vid_path = [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            clip.append(torch.stack(item[0],dim=0)) 

            label.append(item[1])
            vid_path.append(item[2])

    clip = torch.stack(clip, dim=0)

    return clip, label, vid_path

def collate_fn2(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []
    # print(len(batch))
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) 
            label.append(item[1])
            vid_path.append(item[2])
            # frame_list.append(torch.from_numpy(np.asarray(item[3])))
            frame_list.append(torch.from_numpy(np.asarray(list(range(8)))))

    f_clip = torch.stack(f_clip, dim=0)
    frame_list = torch.stack(frame_list, dim=0)
    
    return f_clip, label, vid_path, frame_list 
            
def collate_fn_train(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) 
            label.append(item[1])
            vid_path.append(item[2])
            # frame_list.append(item[3])
            frame_list.append(torch.from_numpy(np.asarray(item[3])))
            # print(len(item[0]))
            # frame_list.append(torch.from_numpy(np.asarray(list(range(len(item[0]))))))
            
    f_clip = torch.stack(f_clip, dim=0)
    frame_list = torch.stack(frame_list, dim=0)

    return f_clip, label, vid_path, frame_list

if __name__ == '__main__':
    import params_linear3_2d3d_crop06 as params
    
    import torchvision
    from PIL import Image, ImageDraw, ImageFont
    
    
    '''visualize = True
    run_id = 'hmdb_try5_MediumCutout'
    dataset = 'hmdb51'
    
    vis_output_path = 'some_visualization/finetuning_dl/' + run_id

    train_dataset = baseline_dataloader_train_strong(params = params, dataset= dataset, shuffle = False, data_percentage = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, collate_fn=collate_fn_train, num_workers=params.num_workers)
   

    print(f'Length of dataset: {len(train_dataset)}')
    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            clip = clip.permute(0,1,3,4,2)
            if params.RGB or params.normalize:
                clip = torch.flip(clip, [4])
            # print(f'Full_clip shape is {clip.shape}')
            # print(f'Label is {label}')
            # print(f'Frame list is {frame_list}')
            # exit()
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
            
            if visualize:
                if dataset == 'ucf101':
                    classes = json.load(open(cfg.class_mapping))['classes']
                elif dataset == 'hmdb51':
                    classes= json.load(open(cfg.hmdb_mapping))
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
                        myFont = ImageFont.truetype('/sensei-fs/users/idave/downloaded_weights/calibri.ttf', 25)   
                        msg = str(frame_list[kk][kk1].item())
                        
                        
                        # print(msg[:-2])
                        # exit()
                        d1.text((100, 0), msg, font=myFont, fill =(255, 255, 255))
                        
                        if kk1 ==0:
                            msg1 = inv_map[label[kk]]
                            d1.text((50, 125), msg1, font=myFont, fill =(255, 255, 255))
                        
                        frame = trans.functional.to_tensor(frame)
                        print(frame.shape)
                        clip1[kk1] = frame.permute(1,2,0)
                    clip1 *= 255
                    clip1 = clip1.to(torch.uint8).permute(0,3,1,2)      
                    image = torchvision.utils.make_grid(clip1, nrow = params.num_frames)
                        # filename =  vis_output_path +'/' + str(counter) + '_' + inv_map[label[counter]]+ '.png' 
                    filename =  vis_output_path +'/' + str(counter) + '.png' 

                    torchvision.io.write_png(image, filename)
                    counter+=1

                exit()
            
    print(f'Time taken to load data is {time.time()-t}')
    print(f'Time taken to load data is {time.time()-t}')'''

    visualize = True
    casia_split = 'doesntmatter'
    run_id = 'ucf101' #'ucf_valTry2_crop1'
    dataset = 'ucf101'#'hmdb51'
    
    vis_output_path = 'some_visualization/finetuning_dl/' + run_id
    
    '''train_dataset = multi_baseline_dataloader_val_strong(params = params,split =1 , dataset=dataset, shuffle = False, data_percentage = 1.0,  mode = 4, cropping_factor=1.0, total_num_modes = 5, casia_split = casia_split)    
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, collate_fn=collate_fn2, num_workers=params.num_workers)'''
    
    train_dataset = baseline_dataloader_train_strong(params = params, dataset= dataset, shuffle = False, data_percentage = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=True, collate_fn=collate_fn_train, num_workers=params.num_workers)

    print(f'Length of dataset: {len(train_dataset)}')
    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            clip = clip.permute(0,1,3,4,2)
            if params.RGB or params.normalize:
                clip = torch.flip(clip, [4])
            # print(f'Full_clip shape is {clip.shape}')
            # print(f'Label is {label}')
            # print(f'Frame list is {frame_list}')
            # exit()
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
            
            if visualize:
                if dataset == 'ucf101':
                    classes = json.load(open(cfg.class_mapping))['classes']
                elif dataset == 'hmdb51':
                    classes= json.load(open(cfg.hmdb_mapping))
                if dataset in ['ucf101','hmdb51']:
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
                        myFont = ImageFont.truetype('/sensei-fs/users/idave/downloaded_weights/calibri.ttf', 25)   
                        msg = str(frame_list[kk][kk1].item())
                        
                        
                        # print(msg[:-2])
                        # exit()
                        d1.text((100, 0), msg, font=myFont, fill =(255, 255, 255))
                        
                        if kk1 ==0:
                            if dataset in ['ucf101','hmdb51']:
                                msg1 = inv_map[label[kk]]
                            else:
                                msg1 = str(label[kk])
                            # else:
                                # msg1 = str(label)
                            d1.text((50, 125), msg1, font=myFont, fill =(255, 255, 255))
                        
                        frame = trans.functional.to_tensor(frame)
                        print(frame.shape)
                        clip1[kk1] = frame.permute(1,2,0)
                    clip1 *= 255
                    clip1 = clip1.to(torch.uint8).permute(0,3,1,2)      
                    image = torchvision.utils.make_grid(clip1, nrow = params.num_frames)
                        # filename =  vis_output_path +'/' + str(counter) + '_' + inv_map[label[counter]]+ '.png' 
                    filename =  vis_output_path +'/' + str(counter) + '.png' 

                    torchvision.io.write_png(image, filename)
                    counter+=1

                exit()
            
    print(f'Time taken to load data is {time.time()-t}')
    # print(f'Time taken to load data is {time.time()-t}')        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

'''if visualize:
    classes = json.load(open(cfg.class_mapping))['classes']
    inv_map = {v: k for k, v in classes.items()}
    if not os.path.exists(vis_output_path):
        os.makedirs(vis_output_path)
    # pickle.dump(clip, open('some_visualization/f_clip.pkl','wb'))
    # pickle.dump(label, open('some_visualization/label.pkl','wb'))
    for counter, single_clip in enumerate(clip):
        frame_id = 0
        os.makedirs(vis_output_path + '/' + str(counter) + '_' + inv_map[label[counter]])

        for single_frame in single_clip:
            # single_frame = single_frame.permute(2,0,1)

            single_frame = single_frame.numpy()
            print(single_frame.shape)

            cv2.imwrite(vis_output_path + '/' + str(counter) + '_' + inv_map[label[counter]] + '/' +str(frame_id) +'.png', single_frame*255)
            frame_id+=1 '''                   
                
    
    

'''train_dataset = multi_baseline_dataloader_val_strong(split =1 , shuffle = False, data_percentage = 1.0,  mode = 2)
train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn2, num_workers=params.num_workers )

print(f'Step involved: {len(train_dataset)/params.batch_size}')
t=time.time()

for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
    if i%25 == 0:
        print()
        # clip = clip.permute(0,1,3,4,2)
        print(f'Full_clip shape is {clip.shape}')
        print(f'Label is {label}')
        # print(f'Frame list is {frame_list}')

        # pickle.dump(clip, open('f_clip.pkl','wb'))
        # pickle.dump(label, open('label.pkl','wb'))
        # exit()
print(f'Time taken to load data is {time.time()-t}')'''

