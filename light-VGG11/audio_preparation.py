from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import pandas as pd
import librosa
import os
import torch
import random
import matplotlib.pyplot as plt
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
  
    
def extract_melspec(file_name):
    audio, sr = librosa.load(file_name)
    n_fft = 2048
    hop_length = 512

    mel_signal = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft, window='hann', n_mels = 88)
    logspec = librosa.power_to_db(mel_signal, ref = 1.0)
    x = logspec
    norm = (x-np.min(x))/(np.max(x)-np.min(x))
    if norm.shape[1] != 87:
       print(file_name)
    #show_spectogram(norm, sr)
    return norm

def show_spectogram(spec, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, x_axis='time',
                            y_axis='mel', sr=sr,
                             ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB', ticks=[0,1])
    #ax.set(title='Mel-frequency spectrogram')
    #plt.savefig('Mel Spectogram.png')
    plt.show()

def generate_dataset(class_1, class_2, test):
    features = []

    # Iterate through each sound file and extract the features
    dir1 = os.listdir(class_1)
    random.Random(1).shuffle(dir1) 
    for i, file in enumerate(dir1):        
        file_name = class_1 + os.fsdecode(file)    
        class_label = 0
        data = extract_melspec(file_name)
        features.append([data, class_label])

    dir2 = os.listdir(class_2)
    random.Random(1).shuffle(dir2)
    for i, file in enumerate(dir2):
        file_name = class_2 + os.fsdecode(file)    
        class_label = 1
        data = extract_melspec(file_name)
        features.append([data, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    x = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = le.fit_transform(y)

    if test:
        x_test = torch.unsqueeze(torch.from_numpy((x)),1)
        y_test = torch.from_numpy(yy)
        file = torch.save([x_test, y_test], "TEST.pt")
        return x_test, y_test
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, yy, test_size=0.25, random_state=1)
        x_train = torch.unsqueeze(torch.from_numpy((x_train)),1)
        x_val = torch.unsqueeze(torch.from_numpy((x_val)),1)
        y_train = torch.from_numpy(y_train)
        y_val = torch.from_numpy(y_val)
        file = torch.save([x_train, x_val, y_train, y_val], "dataset.pt")
        return x_train, y_train, x_val, y_val


import pickle
def extract_files_from_pkl(filename, outname):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        X = data['x']
        y = data['y']
    features = []

    # Iterate through each sound file and extract the features
    for i, data in enumerate(X):        
        class_label = y[i]
        features.append([data, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = le.fit_transform(y)
    x_train, x_val, y_train, y_val = train_test_split(X, yy, test_size=0.25, random_state=1)
    x_train = torch.unsqueeze(torch.from_numpy((x_train)),1)
    x_val = torch.unsqueeze(torch.from_numpy((x_val)),1)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    file = torch.save([x_train, x_val, y_train, y_val], "dataset.pt")
    return x_train, y_train, x_val, y_val


def combine_datasets(dataset1, dataset2, file):
    x_train, x_val, x_test, y_train, y_val, y_test = torch.load(dataset1)
    xt, xv, xtest, yt, yv, ytest = torch.load(dataset2)

    x_train = np.append(xt, x_train.numpy(), axis=0)
    y_train = np.append(yt, y_train.numpy(), axis=0)
    x_val = np.append(xv, x_val.numpy(), axis=0)
    y_val = np.append(yv, y_val.numpy(), axis=0)

    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)

    print(x_train.size(), x_val.size(), x_test.size())
    file = torch.save([x_train, x_val, x_test, y_train, y_val, y_test], file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = int, default=1, help='1 for creating datasets, 0 for combining datasets')
    parser.add_argument('--test_1', type = str, default=r'..\data\TEST\attack\\', help='test folder with positive samples')
    parser.add_argument('--test_0', type = str, default=r'..\data\TEST\normal\\', help='test folder with negative samples')
    parser.add_argument('--train_1', type = str, default=r'..\data\TRAIN\YT\attack\\', help='train folder with positive samples')
    parser.add_argument('--train_0', type = str, default=r'..\data\TRAIN\YT\normal\\', help='train folder with negative samples')
    parser.add_argument('--dataset_name', type = str, default='dataset.pt', help='name of the dataset')
    parser.add_argument('--dataset1', type = str, default='dataset1.pt', help='first dataset to combine')
    parser.add_argument('--dataset2', type = str, default='dataset2.pt', help='second dataset to combine')



    args = parser.parse_args()
    #to create a new dataset
    if args.task:
        x_train, y_train, x_val, y_val  = generate_dataset(args.train_0, args.train_0, 0)
        x_test, y_test = generate_dataset(args.test_0, args.test_1, 1)
        #If the test set stays already exists, instead of generating it again, use the following line
        #y_test, y_test = torch.load("TEST.pt")
        print(x_train.size(), x_val.size(), x_test.size())
        file = torch.save([x_train, x_val, x_test, y_train, y_val, y_test], args.dataset_name)

    #to combine two existing datasets with the same test set
    else:
        combine_datasets(args.dataset1, args.dataset2, args.dataset_name)