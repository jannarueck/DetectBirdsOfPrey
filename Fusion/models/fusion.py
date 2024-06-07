import torch

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

class FusionNet(nn.Module):
    def __init__(self, vgg11, yolov7):
        super(FusionNet, self).__init__()
        self.vgg11= vgg11
        self.yolov7 = yolov7
        self.lin1 = nn.Linear(1408, 128)
        self.lin2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=None)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(23, 40), stride=None)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(46, 80), stride=None)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(92, 160), stride=None)
        #self.maxpool_2 = nn.MaxPool2d(kernel_size=(12, 20), stride=None)
        #self.maxpool_3 = nn.MaxPool2d(kernel_size=(24, 40), stride=None)
        #self.maxpool_4 = nn.MaxPool2d(kernel_size=(48, 80), stride=None)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, img, audio):
        #vgg = self.vgg11(audio).cpu()
        #yolo = self.yolov7(img)[0].cpu()
        out_vgg = get_layer_audio(self.vgg11, audio)
        #print(out_vgg)
        out_yolo76 = get_layer_image(self.yolov7, img,-2, 'act76') #520x20x20
        out_yolo75 = get_layer_image(self.yolov7, img,-3, 'act75') #265x40x40
        out_yolo74 = get_layer_image(self.yolov7, img,-4, 'act74') #128x80x80
        #print(out_yolo76.size(), out_yolo75.size(), out_yolo74.size())

        out_vgg = self.dropout(out_vgg)
        out_yolo76 = self.dropout(out_yolo76)
        out_yolo75 = self.dropout(out_yolo75)
        out_yolo74 = self.dropout(out_yolo74)
        
        out_vgg = self.maxpool_1(out_vgg)
        out_yolo76 = self.maxpool_2(out_yolo76)
        out_yolo75 = self.maxpool_3(out_yolo75)
        out_yolo74 = self.maxpool_4(out_yolo74)


        #print(out_yolo76.size(), out_yolo75.size(), out_yolo74.size())
        x = torch.cat((out_vgg, out_yolo74, out_yolo75, out_yolo76), dim = 1)
        x = torch.squeeze(x)
        #print(x.size())
        x = self.lin1(x)
        x = self.relu(x)
        #print(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)

        return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def get_layer_audio(vgg, x1):
    train_nodes, eval_nodes = get_graph_node_names(vgg)
    #print(train_nodes)
    return_nodes = {
        'features.28':'maxpool'
    }
    model2 = create_feature_extractor(vgg, return_nodes=return_nodes)
    output = model2(x1)
    
    #print("LAYER:",output["maxpool"])

    return output["maxpool"]

def get_layer_image(yolo, x1, layer, name):
    yolo.model[layer].act.register_forward_hook(get_activation(name))
    output = yolo(x1)
    out = activation[name]
    return out
    
    