import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.optim as optim
import time
from Regularization import Regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import platform
import seaborn as sns

from models.experimental import attempt_load
from models.vgg import vgg11_bn
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score
from fusion_dataset import FusionDataset
from torch.utils.data import DataLoader
from conf import settings

from models.fusion import FusionNet
from sklearn.model_selection import train_test_split 
from EarlyStopper import EarlyStopper

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(args, optimizer, i_iter, len_train):
    steps = (len_train/args.b)*150
    lr = lr_poly(args.lr, i_iter, steps, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def train(train_loader, network, optimizer, epoch, loss_function):
    start = time.time()
    network.train()
    train_acc_process = []
    train_loss_process = []
    for batch_index, (images, audio, labels) in enumerate(train_loader):
        print("batch ", batch_index)
        len_train = len(train_loader.dataset)
        step = (len_train/args.b)*(epoch-1)+batch_index #当前的iteration;
        if args.gpu:
            labels = labels.cuda()
            audio = audio.cuda()
            images = images.cuda()
            loss_function = loss_function.cuda()

        optimizer.zero_grad() # clear gradients for this training step
        lr = adjust_learning_rate(args, optimizer, step, len_train)
        outputs = network(images, audio)
        out = outputs.squeeze(dim=-1)
        loss = loss_function(out, labels.float())

        if args.weight_d > 0:
            loss = loss + reg_loss(net)
        
        loss.backward() # backpropogation, compute gradients
        optimizer.step() # apply gradients

        preds = torch.zeros_like(out)
        for i in range(len(out)):
            if out[i] <= 0.5:
                preds[i] = 0
            else:
                preds[i] = 1
        correct_n = preds.eq(labels).sum()
        accuracy_iter = correct_n.float() / len(labels)
        
        if args.gpu:
            accuracy_iter = accuracy_iter.cpu()
        
        train_acc_process.append(accuracy_iter.numpy().tolist())
        train_loss_process.append(loss.item())

    print('Training Epoch: {epoch} [{total_samples}]\tTrain_accuracy: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            np.mean(train_acc_process),
            np.mean(train_loss_process),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            total_samples=len(train_loader.dataset)
    ))
    
    Train_Accuracy.append(np.mean(train_acc_process))
    Train_Loss.append(np.mean(train_loss_process))
    
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    return network

@torch.no_grad()
def eval_training(valid_loader, network,loss_function, epoch=0):

    start = time.time()
    network.eval()
    
    n = 0
    valid_loss = 0.0 # cost function error
    correct = 0.0
    class_target =[]
    class_predict = []

    for (images, audio, labels) in valid_loader:
        if args.gpu:
            images = images.cuda()
            audio = audio.cuda()
            labels = labels.cuda()
            loss_function = loss_function.cuda()

        outputs = network(images, audio)
        out = outputs.squeeze(dim=-1)
        loss = loss_function(out, labels.float())
        valid_loss += loss.item()
        
        preds = torch.zeros_like(out)
        for i in range(len(out)):
            if out[i] <= 0.5:
                preds[i] = 0
            else:
                preds[i] = 1
        correct += preds.eq(labels).sum()
        
        if args.gpu:
            labels = labels.cpu()
            preds = preds.cpu()
        
        class_target.extend(labels.numpy().tolist())
        class_predict.extend(preds.numpy().tolist())
        n +=1
    finish = time.time()
    if args.gpu:
        correct = correct.cpu()
    
    print('Evaluating Network.....')
    print('Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        valid_loss / n, #总的平均loss
        correct.float() / len(valid_loader.dataset),
        finish - start
    ))
    
    #Obtain f1_score of the prediction
    fs = f1_score(class_target, class_predict, average='macro')
    print('f1 score = {}'.format(fs))
    
    #Output the classification report
    print('------------')
    print('Classification Report')
    print(classification_report(class_target, class_predict))
    
    f1_s.append(fs)
    Valid_Loss.append(valid_loss / n)
    Valid_Accuracy.append(correct.float() / len(valid_loader.dataset))
    
    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(args.epoch, args.b, args.lr, args.gpu, args.seed))

    return correct.float() / len(valid_loader.dataset), valid_loss / len(valid_loader.dataset), fs

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--net', type=str, default='canet', help='net type')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')
    parser.add_argument('--b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epoch',type=int, default= 15, help='total training epoches')
    parser.add_argument('--stop_early',type=int, default= 0, help='use early stopping or not')
    parser.add_argument('--seed',type=int, default=1, help='seed')
    parser.add_argument('--weight_d',type=float, default=0.00001, help='weight decay for regularization')
    parser.add_argument('--save_path',type=str, default='setting1', help='saved path of each setting')
    parser.add_argument('--train_path',type=str, default='data/Dataset Combinations/F8.pt', help='saved path of input data')
    parser.add_argument('--test_path',type=str, default='data/TEST/test.pt', help='saved path of input data')
    parser.add_argument('--img_size',type=int, default=1280, help='image size')


    args = parser.parse_args()
    print('CUDA: ' ,torch.cuda.is_available())
    device = 'cpu'
    if args.gpu:
        torch.cuda.manual_seed(args.seed)
        device = 'cuda:0'
      
    else:
        torch.manual_seed(args.seed)
           
    net = vgg11_bn()
    if args.gpu:
        net = net.cuda()

    vgg = net
    vgg.load_state_dict(torch.load("vgg11-best.pth",  map_location=torch.device(device)))
    vgg.eval()

    # Load model
    yolo = attempt_load('yolov7-best.pt', map_location=device)  # load FP32 model

    
    #if half:
        #yolo.half()  # to FP16
    
    #Freeze weights of VGG11 and YOLOv7 net
    for param in vgg.parameters():
        param.requires_grad = False
    
    for param in yolo.parameters():
        param.requires_grad = False

    network = FusionNet(vgg, yolo)
    if args.gpu:
        network = network.cuda()

    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}'.format(args.epoch, args.b, args.lr, args.gpu))

    sysstr = platform.system()
    if(sysstr =="Windows"):
        num_workers = 0
    else:
        num_workers = 8

    train_pathway = args.train_path
    test_pathway = args.test_path

    
    images, audio, labels = torch.load(train_pathway)
    img_train, img_val, audio_train, audio_val, y_train, y_val = train_test_split(images, audio, labels, test_size=0.25, random_state=2)
    train_data = FusionDataset(1, img_train, audio_train, y_train, args.img_size)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=num_workers, batch_size=args.b)
    print("Train Data loaded")
    val_data = FusionDataset(2, img_val, audio_val, y_val, args.img_size)
    valid_loader = DataLoader(val_data, shuffle=True, num_workers=num_workers, batch_size=args.b)
    print("Validation Data loaded")

    img_test, audio_test, y_test = torch.load(test_pathway)
    test_data = FusionDataset(3, img_test, audio_test, y_test, args.img_size)
    test_loader = DataLoader(test_data, shuffle=True, num_workers=num_workers, batch_size=args.b)

    if args.weight_d > 0:
        reg_loss=Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")
        
    # TRAINING AND VALIDATION
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=5e-4)


    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.save_path, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
  
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path_pth = os.path.join(checkpoint_path, 'fusion-{type}.pth')

    best_acc = 0.0
    Train_Loss = []
    Train_Accuracy = []
    Valid_Loss = []
    Valid_Accuracy = []
    f1_s = []
    best_epoch = 1
    best_weights_path = checkpoint_path_pth.format(type='best')

    early_stopper = EarlyStopper(patience=3, min_delta=0.005)
    for epoch in range(1, args.epoch + 1):  
        print("Epoch: ", epoch) 
        net = train(train_loader, network, optimizer, epoch, loss_function)
        acc, validation_loss, fs_valid = eval_training(valid_loader, net, loss_function, epoch)

        #start to save best performance model (according to the accuracy on validation dataset) after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0] and best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)
        if early_stopper.early_stop(acc.item() and args.stop_early):
            print("Stop early at epoch ", epoch)             
            break

    print('best epoch is {}'.format(best_epoch))
    #####Output results
    #plot train loss and accuracy vary over time
    font_1 = {'weight' : 'normal', 'size'   : 20}
    fig1=plt.figure(figsize=(12,9))
    plt.title('Accuracy')
    index_train = list(range(1,len(Train_Accuracy)+1))
    weight = 0.6
    last = Train_Accuracy[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in Train_Accuracy:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    
    last = Valid_Accuracy[0]  # First value in the plot (first timestep)
    smoothed_v = list()
    for point in Valid_Accuracy:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed_v.append(smoothed_val)                        # Save it
        last = smoothed_val  
    plt.plot(index_train,smoothed,color='skyblue',label='train_accuracy')
    plt.plot(index_train,smoothed_v,color='red',label='valid_accuracy')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,args.epoch)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Accuracy',font_1)
    
    acc_figuresavedpath = os.path.join(checkpoint_path,'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath)
    # plt.show()
    
    #plot valid loss and accuracy vary over time
    fig2=plt.figure(figsize=(12,9))
    plt.title('Loss')
    index_valid = list(range(1,len(Valid_Loss)+1))
    weight = 0.6
    last = Train_Loss[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in Train_Loss:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    
    last = Valid_Loss[0]  # First value in the plot (first timestep)
    smoothed_v = list()
    for point in Valid_Loss:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed_v.append(smoothed_val)                        # Save it
        last = smoothed_val 

    plt.plot(index_valid,smoothed,color='skyblue', label='train_loss')
    plt.plot(index_valid,smoothed_v,color='red', label='valid_loss')
    # plt.legend(prop=font_2)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,args.epoch)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)
    
    loss_figuresavedpath = os.path.join(checkpoint_path,'Loss_curve.png')
    plt.savefig(loss_figuresavedpath)
    # plt.show()
    
    #plot f1 score varying over time
    fig3=plt.figure(figsize=(12,9))
    plt.title('F1-score',font_1)
    index_fs = list(range(1,len(f1_s)+1))
    plt.plot(index_fs,f1_s,color='skyblue')
    # plt.legend(prop=font_2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    fs_figuresavedpath = os.path.join(checkpoint_path,'F1-score.png')
    plt.savefig(fs_figuresavedpath)
    # TESTING
    out_txtsavedpath = os.path.join(checkpoint_path,'output.txt')
    f = open(out_txtsavedpath, 'w+')
    
    print('File:{}, Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Saved path: {}'.format(
        args.train_path, args.seed, args.epoch, args.b, args.lr, 0, args.gpu, args.save_path),
        file=f)
    
    print('index: {}; maximum value of validation accuracy: {}.'.format(Valid_Accuracy.index(max(Valid_Accuracy))+1, max(Valid_Accuracy)), file=f)
    print('index: {}; maximum value of validation f1-score: {}.'.format(f1_s.index(max(f1_s))+1, max(f1_s)), file=f)
    print('--------------------------------------------------')
    print('Validation accuracy: {}'.format(Valid_Accuracy), file=f)
    print('Validation F1-score: {}'.format(f1_s), file=f)

    #Load the best trained model and test testing data
    best_net = FusionNet(vgg, yolo)
    best_net = best_net.cuda()
    #print(best_weights_path)
    best_net.load_state_dict(torch.load(best_weights_path))
    #print(best_net)

    best_net.eval()
    number = 0
    correct_test = 0.0
    test_target =[]
    test_predict = []
    wrong_cl = []
    
    with torch.no_grad():
        
        start = time.time()
        
        for n_iter, (image, audio, labels, path) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                audio = audio.cuda()
                labels = labels.cuda()

            t0 = time.time()
            output = best_net(image, audio)
            out = output.squeeze(dim=-1)
            t1 = time.time()

            preds = torch.zeros_like(out)
            
            for i in range(len(out)):
                if out[i] <= 0.5:
                    preds[i] = 0
                else:
                    preds[i] = 1

            correct_test += preds.eq(labels).sum()
            
            if args.gpu:
                labels = labels.cpu()
                preds = preds.cpu()

            test_target.extend(labels.numpy().tolist())
            test_predict.extend(preds.numpy().tolist())
        
            number +=1      
        
        print('Label values: {}'.format(test_target), file=f)
        print('Predicted values: {}'.format(test_predict), file=f)

        finish = time.time()
        accuracy_test = correct_test.float() / len(test_loader.dataset)
        print('Testing network......', file=f)
        print('Test set: Accuracy: {:.5f}, Time consumed: {:.5f}s'.format(
            accuracy_test,
            finish - start
            ), file=f)
        
        #Obtain f1_score of the prediction
        fs_test = f1_score(test_target, test_predict, average='macro')
        print('f1 score = {:.5f}'.format(fs_test), file=f)
        
        kappa_value = cohen_kappa_score(test_target, test_predict)
        print("kappa value = {:.5f}".format(kappa_value), file=f)
        
        precision_test = precision_score(test_target, test_predict, average='macro')
        print('precision = {:.5f}'.format(precision_test), file=f)
        
        recall_test = recall_score(test_target, test_predict, average='macro')
        print('recall = {:.5f}'.format(recall_test), file=f)
        
        #Output the classification report
        print('------------', file=f)
        print('Classification Report', file=f)
        print(classification_report(test_target, test_predict), file=f)
        Class_labels = ['0','1']

        #Show the confusion matrix so that it can help us observe the results more intuitively
        def show_confusion_matrix(validations, predictions):
            matrix = confusion_matrix(validations, predictions) #No one-hot
            #matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix,
                  cmap="coolwarm",
                  linecolor='white',
                  linewidths=1,
                  xticklabels=Class_labels,
                  yticklabels=Class_labels,
                  annot=True,
                  fmt="d")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_figuresavedpath = os.path.join(checkpoint_path,'Confusion_matrix.png')
            plt.savefig(cm_figuresavedpath)

        show_confusion_matrix(test_target, test_predict)
        
    
    
    
    