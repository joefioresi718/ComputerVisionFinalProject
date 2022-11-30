import argparse
import itertools
import numpy as np
import os
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import warnings

from model_loaders import *
import parameters as params
import config as cfg
from ucf101_dl import *
from dl_linear_frameids import *


# Get rid of DepreciationWarnings.
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, data_loader, model, criterion, optimizer, writer, use_cuda, lr):
    print(f'Train at epoch {epoch}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        writer.add_scalar('Learning Rate', lr, epoch)  
        print(f'Learning rate is: {param_group["lr"]}')
  
    losses = []

    # Set model to train.
    model.train()

    for i, (videos, label, _, _) in enumerate(data_loader):
        optimizer.zero_grad()

        if use_cuda:
            videos = videos.cuda()
            label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).cuda()

        # Reshape UCF101 inputs.
        output = model(videos.permute(0, 2, 1, 3, 4))

        # Compute loss.
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 5 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
    
    print(f'Training Epoch: {epoch}, Loss: {np.mean(losses)}')
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, videos, output, label

    return model, np.mean(losses)

def val_epoch(run_id, epoch,mode,cropping_fac, pred_dict,label_dict, data_loader, model, criterion, writer, use_cuda,device_name):
    print(f'validation at epoch {epoch} - mode {mode},  cropping_fac {cropping_fac}  ')
    
    model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, frameids) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        if len(inputs.shape) != 1:
            inputs = inputs.permute(0,2,1,3,4)
            if params.RGB or params.normalize:
                inputs = torch.flip(inputs, [1]) #TODO: lookinto pretrained weights requirement-pixel range, normalization
            # print('convert this input to rgb', inputs.shape) #[8, 3, 16, 224, 224]) #torch.Size([56, 3, 16, 112, 112])
            # exit()


            if params.normalize:
                # print('convert this input to normalized', inputs.shape) #[8, 3, 16, 224, 224])           

                inputs = inputs.permute(0,2,1,3,4)

                inputs_shape = inputs.shape
                inputs = inputs.reshape(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4])
                inputs = torchvision.transforms.functional.normalize(inputs, mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                inputs = inputs.reshape(inputs_shape)
            
                inputs = inputs.permute(0,2,1,3,4)
            if use_cuda:
                inputs = inputs.to(device=torch.device(device_name))
                label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).to(device=torch.device(device_name))
                # frameids = frameids.to(device=torch.device(device_name))
                frameids = torch.arange(0, params.num_frames, 1).to(torch.int).repeat(inputs.shape[0], 1).cuda() #frameids.cuda().to(torch.int)

                        # if use_cuda:
#             videos = videos.cuda()
#             label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).cuda()

            with torch.no_grad():
                output = model(inputs)
                # Compute loss.
                loss = criterion(output, label)

            losses.append(loss.item())


            predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())


            if i+1 % 45 == 0:
                print("Validation Epoch ", epoch , "mode", mode, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:,0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    print_pred_array = []

    
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    # print(f'Correct Count is {correct_count}')
    print(f'Epoch {epoch}, mode {mode}, cropping_fac {cropping_fac} Accuracy: {accuracy*100 :.3f}')
    return pred_dict, label_dict, accuracy, np.mean(losses)


# Validation epoch.
# def val_epoch(epoch, data_loader, model, criterion, use_cuda):
#     print(f'Validation at epoch {epoch}.')
    
#     # Set model to evaluation mode.
#     model.eval()

#     losses = []
#     predictions = []
    
#     num_processed_samples = 0
#     # Group and aggregate output of a video
#     num_videos = len(data_loader) * params.batch_size_ucf101
#     num_classes = params.num_classes
#     agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device='cuda')
#     agg_targets = torch.zeros((num_videos), dtype=torch.int32, device='cuda')

#     for i, (videos, label, _, video_idx) in enumerate(data_loader):
#         # vid_paths.extend(vid_path)
#         # ground_truth.extend(label)
#         if use_cuda:
#             videos = videos.cuda()
#             label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).cuda()

#         with torch.no_grad():
#             output = model(videos.permute(0, 2, 1, 3, 4))
#             # Compute loss.
#             loss = criterion(output, label)

#         losses.append(loss.item())
#         # Generate prediction.
#         # Use softmax to convert output into prediction probability
#         # preds = torch.softmax(output, dim=1)
#         # for b in range(videos.size(0)):
#         #     idx = video_idx[b].item()
#         #     agg_preds[idx] += preds[b].detach()
#         #     agg_targets[idx] = label[b].detach().item()
#         # acc1, acc5 = accuracy(output, label, topk=(1, 5))
#         # num_processed_samples += videos.shape[0]
#         predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())

#         if i % 5 == 0:
#             print(f'Validation Epoch {epoch}, Batch {i} - Loss : {np.mean(losses)}')
        
#     del videos, output, label, loss 

#     # Reduce the agg_preds and agg_targets from all gpu and show result
#     acc1, acc5 = accuracy(agg_preds, agg_targets, topk=(1, 5))
#     print(f'Total video count: {num_processed_samples}')
#     print(f'Epoch {epoch}, * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}, Loss: {np.mean(losses)}')

#     return acc1, acc5, np.mean(losses)


# Utils accuracy function, from: https://github.com/pytorch/vision/blob/main/references/video_classification/utils.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
    

# Main code. 
def train_classifier(run_id, arch, saved_model):
    for k, v in params.__dict__.items():
        print(f'{k} : {v}')
    # Empty cuda cache.
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load in model.
    model = load_ft_model(arch=arch, kin_pretrained=True, saved_model_file=saved_model)

    # Init loss function.
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        print(f'Multiple GPUS found!')
        model = nn.DataParallel(model)
        model.cuda()
        criterion.cuda()
    else:
        print('Only 1 GPU is available')
        model.cuda()
        criterion.cuda()

    # Check if model params need to be summed.
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    import params_linear3_2d3d_crop06 as params1
    # train_dataset = ucf101_ar_dataset(data_split='train', shuffle=False, data_percentage=params.data_percentage_ucf101)
    train_dataset = baseline_dataloader_train_strong(params=params1, dataset='ucf101', data_percentage=params.data_percentage_ucf101)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size_ucf101, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size_ucf101}')

    # validation_dataset = ucf101_ar_dataset(data_split='test', shuffle=True, data_percentage=params.data_percentage_ucf101)
    validation_dataset = multi_baseline_dataloader_val_strong(params=params1, dataset='ucf101', shuffle=False, data_percentage=params.data_percentage_ucf101)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)

    print(f'Validation dataset length: {len(validation_dataset)}')
    print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')

    val_array = [1, 3, 5, 10, 20, 25, 30, 35] + [40 + x for x in range(100)]

    skip = list(range(1, params.num_skips + 1))
    print(f'Num skips {skip}')
    print(f'Base learning rate {params.learning_rate}')
    print(f'Scheduler patient {params.lr_patience}')
    print(f'Scheduler drop {params.scheduled_drop}')

    accuracy = 0
    lr_flag1 = 0
    lr_counter = 0
    best_score = 0
    train_best = 1000
    epoch1 = 1

    modes = list(range(params.num_modes))
    cropping_facs = params.cropping_fac1

    learning_rate = params.learning_rate
    for epoch in range(epoch1, params.num_epochs + 1):
        if epoch < params.warmup and lr_flag1 == 0:
            learning_rate = params.warmup_array[epoch] * params.learning_rate

        print(f'Epoch {epoch} started')
        start = time.time()

        model, train_loss = train_epoch(epoch, train_dataloader, model, criterion, optimizer, writer, use_cuda, learning_rate)
        
        # Used for lr scheduler.
        if train_loss > train_best:
            lr_counter += 1
        
        if lr_counter > params.lr_patience:
            lr_counter = 0
            learning_rate = learning_rate/params.scheduled_drop
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f'Learning rate dropping to its {params.scheduled_drop}th value to {learning_rate} at epoch {epoch}')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # Best results.
        if train_best > train_loss:
            train_best = train_loss

        # Validation epoch.
        if epoch in val_array:
            # acc1, acc5, loss = val_epoch(epoch, validation_dataloader, model, criterion, use_cuda)
            # writer.add_scalar('Validation Acc@1', acc1, epoch)
            # writer.add_scalar('Validation Acc@5', acc5, epoch)
            device_name = 'cuda'
            pred_dict = {}
            label_dict = {}
            val_losses =[]
            for val_iter in range(len(modes)):
                for cropping_fac in cropping_facs:
                    # try:
                    validation_dataset = multi_baseline_dataloader_val_strong(params = params1, dataset='ucf101', shuffle = True, data_percentage = params.data_percentage_ucf101,\
                        mode = modes[val_iter], cropping_factor= cropping_fac, total_num_modes = params.num_modes, casia_split = params.casia_split)
                    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                    pred_dict, label_dict, accuracy, loss = val_epoch(run_id, epoch, modes[val_iter], cropping_fac, pred_dict, label_dict, validation_dataloader, model, criterion, writer, use_cuda,device_name)
                    val_losses.append(loss)

                    predictions1 = np.zeros((len(list(pred_dict.keys())),params.num_classes))
                    ground_truth1 = []
                    entry = 0
                    for key in pred_dict.keys():
                        predictions1[entry] = np.mean(pred_dict[key], axis =0)
                        entry+=1

                    for key in label_dict.keys():
                        ground_truth1.append(label_dict[key])

                    pred_array1 = np.flip(np.argsort(predictions1,axis=1),axis=1) # Prediction with the most confidence is the first element here
                    c_pred1 = pred_array1[:,0]

                    correct_count1 = np.sum(c_pred1==ground_truth1)
                    accuracy11 = float(correct_count1)/len(c_pred1)


                    print(f'Running Avg Accuracy is for epoch {epoch}, mode {modes[val_iter]}, is {accuracy11*100 :.3f}% ')  
                    # if acc1 > best_score:
                    #     print('++++++++++++++++++++++++++++++')
                    #     print(f'Epoch {epoch} is the best model till now for {run_id}!')
                    #     print('++++++++++++++++++++++++++++++')
                    #     save_dir = os.path.join(cfg.saved_models_dir, run_id)
                    #     if not os.path.exists(save_dir):
                    #         os.makedirs(save_dir)
                    #     save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(acc1)[:6]}.pth')
                    #     states = {
                    #         'epoch': epoch + 1,
                    #         'lr_counter': lr_counter,
                    #         'model_state_dict': model.state_dict(),
                    #         'optimizer': optimizer.state_dict()
                    #     }
                    #     torch.save(states, save_file_path)
                    #     best_score = accuracy
                    # except:
                    #     print('not working')

        # Temp saving.
        save_dir = os.path.join(cfg.saved_models_dir, run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'lr_counter' : lr_counter,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        if learning_rate < 1e-12:
            print(f'Learning rate is very low now, stopping the training.')
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default='default_ft_baseline', help='run_id')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None, help='saved_model')
    parser.add_argument("--arch", dest='arch', type=str, required=False, default='r3d', help='model architecture')

    args = parser.parse_args()
    print(f'Run ID: {args.run_id}')
    print(f'Architecture: {args.arch}')

    train_classifier(args.run_id, args.arch, args.saved_model)
