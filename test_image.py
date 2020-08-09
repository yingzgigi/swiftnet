import argparse
from pathlib import Path
import importlib.util
from PIL import Image as pimg
import skimage.transform

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import numpy as np

from data.ade20k import *
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import *
from data.transform import *
from evaluation.evaluate import *
from evaluation import StorePreds

def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--rgb', default=None, metavar='DIR', help='path to image')
parser.add_argument('--depth', default=None, metavar='DIR', help='path to image')
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
    
    #load_ckpt(model, None, args.last_ckpt, device)#
    model.eval()#
    #model.to(device)#
    
    #depth = pimg.open(args.depth)
    image = pimg.open(args.rgb)#
    image = np.array(image, np.float32)
    
    image = skimage.transform.resize(image, (1024, 2048), order = 1, mode='reflect', preserve_range=True)
    #depth = skimage.transform.resize(depth, (1024, 2048), order=0, mode='reflect', preserve_range=True)
    
    
    if len(image.shape) == 3:
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    #depth = np.array(depth, np.uint8)
    #depth = torch.from_numpy(depth)
    
    
    model.load_state_dict(torch.load('weights/rn18_single_scale/model_best.pt'))
    
    #loader_pred = DataLoader(pred_image, batch_size=1, collate_fn=custom_collate)#from rn18_single_scale
    
    #params
    conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)

    logits, additional = model.forward(image, (1024, 2048), (1024,2048))
    pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)###
    
    class_info, color_info = ade20k.init_ade20k_class_color_info(Path('/home/hchen/Documents/yzh/swiftnet2/swiftnet/datasets'))
    #print(class_info, color_info)
    
    store_dir =  f'test_image/'
    to_color = ColorizeLabels(color_info)
    to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])

    #eval_observers = [StorePreds(store_dir, to_image, to_color)] #store colorized image
    
    store_img = np.concatenate([i.astype(np.uint8) for i in to_color(pred)], axis=0)
    print(store_img)
    store_img = pimg.fromarray(store_img)
    store_img.thumbnail((960, 1344))
    store_img.save(f'predict1.jpg')
    
    #pred = get_pred(logits, class_info, conf_mat)
    #print(pred)
    #class_info = conf.dataset_val.class_info
    
    #model = conf.model.cuda()
    
    #pred_semseg(model, loader, class_info) #need loader for test image
'''    
def pred_semseg(model, data_loader, class_info, observers=()):
    logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
    pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)###
'''