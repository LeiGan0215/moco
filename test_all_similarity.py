import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler

import moco.loader
import moco.builder

import os
from PIL import Image
import numpy as np
import json

def build_model(arch='resnet50', gpu_id=7, dim=128):
    # build model
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch](num_classes=dim)
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)
    model.eval()
    # for name in model.state_dict():
    #     print(name)
    # debug
    # load model weight
    # Map model to be loaded to specified single gpu.
    model_path = r'/data2/ganlei/code/moco/save_res/semi_similarity/1/checkpoint_0199.pth.tar'
    loc = 'cuda:{}'.format(gpu_id)
    checkpoint = torch.load(model_path, map_location=loc)
    """
    checkpoint['state_dict']
    module.encoder_q.fc.weight
    module.encoder_q.fc.bias
    module.encoder_k.fc.weight                                                                                                   │+-------------------------------+----------------------+----------------------+
    module.encoder_k.fc.bias
    """
    # del module
    # del param of k
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    epoch = checkpoint["epoch"]
    model.load_state_dict(state_dict)
    return model

def img2tensor(img, gpu_id=7):
    """
    to_tensor, norm, cuda
    """
    to_tensor = transforms.Compose([
        transforms.Resize(224, ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小
    img_tensor = img.cuda(gpu_id)
    return img_tensor

def read_img(img_path):
    """
    read img, resize, img2tensor
    """
    img = Image.open(img_path)
    # img = img.resize((224, 224), Image.BILINEAR) # resize为统一大小
    # img = np.array(img)
    img = img.convert('RGB')
    # img to tensor
    img_tensor = img2tensor(img)
    return img_tensor

def inference_single_encoder(img_tensor, model):
    return model(img_tensor)

def cal_similarity(x1, x2, dim=1, eps=1e-8):
    """
    cos similarity of two feature
    """
    return F.cosine_similarity(x1, x2, dim, eps)

def get_num(path):
    num=0
    for file in os.listdir(path):
        num+=1
    print("{} has {} imgs!".format(path, num))
    return num

if __name__ == '__main__':
    model = build_model()

    path_1 = r'/data2/ganlei/datasets/image_plus/train/360'
    path_2 = r'/data2/ganlei/datasets/image_plus/train/360'

    # m = get_num(path_1)
    # n = get_num(path_2)

    similarity_matrix = dict()

    for file_1 in os.listdir(path_1):
        img1_tensor = read_img(os.path.join(path_1, file_1))
        feature_1 = inference_single_encoder(img1_tensor, model)  # []
        for file_2 in os.listdir(path_2):
            img2_tensor = read_img(os.path.join(path_2, file_2))
            feature_2 = inference_single_encoder(img2_tensor, model) # [1, 128]
            simlarity = cal_similarity(feature_1, feature_2)
            key=file_1+','+file_2
            similarity_matrix[key]=simlarity.item()
            print(key, ": ", simlarity)

    json_str = json.dumps(similarity_matrix, indent=4)
    with open('./similarity_matrix/similarity_matrix.json', 'w') as json_file:
        json_file.write(json_str)



