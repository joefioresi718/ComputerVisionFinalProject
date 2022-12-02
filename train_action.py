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


# Get rid of DepreciationWarnings.
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Find optimal algorithms for the hardware.
torch.backends.cudnn.benchmark = True


# Training epoch.
def train_epoch(epoch, data_loader, model, criterion, optimizer, writer, use_cuda, lr, arch):
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
        if arch != 'vivit':
            videos = videos.permute(0, 2, 1, 3, 4)
        output = model(videos)

        # Compute loss.
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 50 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
    
    print(f'Training Epoch: {epoch}, Loss: {np.mean(losses)}')
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, videos, output, label

    return model, np.mean(losses)


# Validation epoch.
def val_epoch(epoch, data_loader, model, criterion, use_cuda, arch):
    print(f'Validation at epoch {epoch}.')
    
    # Set model to evaluation mode.
    model.eval()

    losses = []
    
    num_processed_samples = 0
    # Group and aggregate output of a video
    num_videos = len(data_loader) * params.v_batch_size
    num_classes = params.num_classes
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device='cuda')
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device='cuda')

    for i, (videos, label, _, video_idx) in enumerate(data_loader):
        if use_cuda:
            videos = videos.cuda()
            label = torch.from_numpy(np.asarray(label)).type(torch.LongTensor).cuda()

        with torch.no_grad():
            # Reshape UCF101 inputs.
            if arch != 'vivit':
                videos = videos.permute(0, 2, 1, 3, 4)
            output = model(videos)
            # Compute loss.
            loss = criterion(output, label)

        losses.append(loss.item())
        # Generate prediction.
        # Use softmax to convert output into prediction probability
        preds = torch.softmax(output, dim=1)
        for b in range(videos.size(0)):
            idx = video_idx[b].item()
            agg_preds[idx] += preds[b].detach()
            agg_targets[idx] = label[b].detach().item()
        num_processed_samples += videos.shape[0]

        if i % 50 == 0:
            print(f'Validation Epoch {epoch}, Batch {i} - Loss : {np.mean(losses)}', flush=True)
        
    del videos, output, label, loss 

    acc1, acc5 = accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(f'Total video count: {num_processed_samples}')
    print(f'Epoch {epoch}, * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}, Loss: {np.mean(losses)}')

    return acc1, acc5, np.mean(losses)


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
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
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
    train_dataset = ucf101_ar_train_dataset(params=params, shuffle=True, data_percentage=params.data_percentage)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')

    validation_dataset = ucf101_ar_val_dataset(params=params, shuffle=True, data_percentage=params.data_percentage)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers)

    print(f'Validation dataset length: {len(validation_dataset)}')
    print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')

    val_array = [1, 3, 5, 10, 20, 25, 30, 35] + [40 + x for x in range(100)]

    skip = list(range(1, params.num_skips + 1))
    print(f'Num skips {skip}')
    print(f'Base learning rate {params.learning_rate}')
    print(f'Scheduler patient {params.lr_patience}')
    print(f'Scheduler drop {params.scheduled_drop}')

    lr_flag1 = 0
    lr_counter = 0
    best_score = 0
    train_best = 1000
    epoch1 = 1

    learning_rate = params.learning_rate
    for epoch in range(epoch1, params.num_epochs + 1):
        if epoch < params.warmup and lr_flag1 == 0:
            learning_rate = params.warmup_array[epoch] * params.learning_rate

        print(f'Epoch {epoch} started')
        start = time.time()

        model, train_loss = train_epoch(epoch, train_dataloader, model, criterion, optimizer, writer, use_cuda, learning_rate, arch)
        
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
            acc1, acc5, loss = val_epoch(epoch, validation_dataloader, model, criterion, use_cuda, arch)
            writer.add_scalar('Validation Acc@1', acc1, epoch)
            writer.add_scalar('Validation Acc@5', acc5, epoch)

            if acc1 > best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, f'model_{epoch}_bestAcc_{str(acc1)[:6]}.pth')
                states = {
                    'epoch': epoch + 1,
                    'lr_counter': lr_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)
                best_score = acc1

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
