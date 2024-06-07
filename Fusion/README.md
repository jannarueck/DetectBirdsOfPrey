# Fusion Network


## Prepare Input

To prepare video files for training/testing the fusion network follow the following steps

1. run fusion_data_generation.py as follows to extract frames and coherent audio samples from video file 

``` shell
python fusion_data_generation --video 'video.mp4' --audio_f 'video_name/audio/' --img_f 'video_name/img/'

```
the input video should be in .mp4 format

2. run fusion_data_preparation.py as follows to extract Mel spectograms from samples and save them with their coherent image + label as Train/Val set or Test set in .pt file

``` shell
python fusion_data_generation --task 1 --audio_f 'data/audio/' --img_f 'data/audio/' --thresh 300 --fname 'dataset.pt'
```

3. Optional: run fusion_data_preparation.py as follows to combine existing datasets

``` shell
python fusion_data_generation --task 0 --fname 'dataset.pt' --dataset1 'datset1.pt' --dataset2 'dataset2.pt'
```

## Training and Testing

Run the train_fusion.py file to train and test the model, adjust the parameters or use the default settings below

``` shell
python train_fusion.py --gpu 1 --b 64 --lr 0.0001 --epoch 20 --stop_early 0 --seed 1 -weight_d 0.00001 --save_path 'setting1' --train_path 'data/C1_H2_YT1.pt' --test_path 'data/test.pt' --img_size 1280
```

the results are saved in the folder checkpoint