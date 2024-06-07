# light-VGG11


## Prepare Input

To prepare audio files for training/testing the light-VGG11 follow the following steps

1. run audio_generation.py as follows to extract and save 2 second non-overlapping audio chunks from file

``` shell
python audio_generation --mp3 0 --mp4 1 --split 0 --start 0 --end 0 --input_data 'video.mp4' --chunk_name --save_folder './alarm_calls'
```
the input file can be in the form of mp3, mp4 or wav and the path is declares using --input_data
mp4 or mp4 files must be converted to wav files by setting the options --mp3 or --mp4 to 1
a certain portion of the audio file can be extracted by setting --split option to one and defining the start and end of the portion in ms using --start and --end
2 ms audio chunks are exported to the folder --save_folder with the name --chunk_name + an index

2. run audio_preparation.py as follows to extract Mel spectograms of each chunk and save them with their label as a Train/Val/Test split in .pt file
``` shell
python --task 1 --test_1 'data/test1//' --test_0 'data/test0//' --train_1 'data/train1//' --train_0 'data/test0//' --dataset_name 'dataset.pt'

```
add the folder name for the positive (--test1) and negative test samples (--test0)
add the folder name for the positive (--train1) and negative train samples (--train0)

3. Optional: run audio_preparation.py as follows to combine two existing datasets

``` shell
python --task 0 --dataset_name 'dataset.pt' --dataset1 'dataset1.pt' --dataset2 'dataset2.pt'
```

The resulting .pt can be directly fed into the light-VGG11 model

## Training and Testing

Run the train.py file to train and test the light-VGG11 model, adjust the parameters or use the default settings below

``` shell
python train.py --net 'vgg11' --gpu 1 --b 32 --lr 0.0001 --epoch 50 --seed 1 --weight_d 0.00001 --save_path 'setting1' --data_path 'dataset.pt'

```
The results are saved in the folder checkpoint

