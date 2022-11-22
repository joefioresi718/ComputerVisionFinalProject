import os.path
import torch
import torch.nn as nn
from torchvision.models import swin_transformer, Swin_B_Weights

import parameters as params
from models.r3d import r3d_18_classifier
from models.vivit import ViViT
from models.modeling_finetune import vit_base_patch16_224
from models.i3d import InceptionI3d


# Ft model loading function.
def load_ft_model(arch='r3d', saved_model_file=None, num_classes=params.num_classes, kin_pretrained=False):
    if arch == 'r3d':
        ft_model = build_r3d_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'i3d':
        ft_model = build_i3d_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'videoMAE':
        ft_model = build_videoMAE_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    elif arch == 'vivit':
        ft_model = build_vivit_classifier(num_classes=num_classes, pretrained=kin_pretrained)
    else:
        print(f'Architecture {arch} invalid for ft_model. Try \'r3d\', \'i3d\', \'videoMAE\', or \'vivit\'')
        return

    # Load in saved model.
    if saved_model_file:
        saved_dict = torch.load(saved_model_file)
        try:
            model.load_state_dict(saved_dict['ft_model_state_dict'], strict=True)
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in saved_dict['ft_model_state_dict'].items():
                name = k[7:]  # Remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
        print(f'ft_model loaded from {saved_model_file} successsfully!')
    else:
        print(f'ft_model freshly initialized! Pretrained: {kin_pretrained}')

    return ft_model


# Create unweighted video classifification transformer.
# TODO: implement ViViT pretraining.
def build_vivit_classifier(num_classes=params.num_classes, pretrained=True):
    model = ViViT(params.reso_h, params.patch_size, num_classes, num_frames=params.num_frames)
    return model


# Create video MAE classification transformer.
def build_videoMAE_classifier(num_classes=params.num_classes, pretrained=True):
    model = vit_base_patch16_224(num_classes=num_classes)
    if pretrained:
        checkpoint_model = torch.load(os.path.join('saved_models', 'mae_checkpoint.pth'))['module']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)
    return model


# Load default pretrained action recognition model.
def build_r3d_classifier(num_classes=params.num_classes, pretrained=False):
    model = r3d_18_classifier(pretrained=pretrained, progress=False)
    # model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1), dilation=(2,1,1), bias=False)
    # model.layer4[0].downsample[0] = nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model 


# Build I3D action recognition model.
def build_i3d_classifier(num_classes=params.num_classes, pretrained=True):
    if pretrained:
        num_classes = 400
    model = InceptionI3d(num_classes=num_classes, dropout_keep_prob=0.5)
    if pretrained:
        saved_weights = torch.load(os.path.join('saved_models', 'rgb_imagenet.pt'))
        model.load_state_dict(saved_weights, strict=True)
    if params.num_classes != 400:
        model.replace_logits(params.num_classes)
    return model


if __name__ == '__main__':
    inputs = torch.rand((1, 3, 16, 224, 224))
    model = load_ft_model(arch='i3d', kin_pretrained=True)   
    with torch.no_grad():
        output = model(inputs)
        # features = model.extract_features(inputs)
    
    print(output)
    print(f'Output shape is: {output.shape}')

    # print(features.squeeze().cpu().numpy())
    # print(f'Feature shape is: {features.shape}')
    


