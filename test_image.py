import argparse
from pathlib import Path
import importlib.util

import torch
from torch.utils.data import DataLoader
import numpy as np

from data.ade20k import *
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import *
from data.transform import *
from evaluation.evaluate import *

def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--rgb', default=None, metavar='DIR', help='path to image')
parser.add_argument('--output', default=None, metavar='DIR', help='path to output')
parser.add_argument('--profile', dest='profile', action='store_true', help='Profile one forward pass')


if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config) #load network model
    pred_image = Path(args.rgb)
    
    #class_info, color_info = init_ade20k_class_color_info(Path('/home/hchen/Documents/yzh/swiftnet2/swiftnet/datasets'))
    num_classes = 150
    
    scale = 1
    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]
    resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)#####4
    model = SemsegModel(resnet, num_classes) ####3
    
    model.load_state_dict(torch.load('weights/rn18_single_scale/model_best.pt'))
    
    loader_pred = DataLoader(pred_image, batch_size=1, collate_fn=custom_collate)#from rn18_single_scale
    
    #params
    conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)

    logits, additional = model.do_forward(1, loader_pred.shape[1:3])
    pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)###
    
    pred = get_pred(logits, class_info, conf_mat)
    print(pred)
    #class_info = conf.dataset_val.class_info
    
    #model = conf.model.cuda()
    
    #pred_semseg(model, loader, class_info) #need loader for test image
'''    
def pred_semseg(model, data_loader, class_info, observers=()):
    logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
    pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)###
'''