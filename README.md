# Detect Birds Of Prey
This work proposes a multimodal learning approach to detects birds of prey in free-range poultry systems. 
The developed network employs two modalities: the detection of birds of prey on images and the detection of the chickens' alarm calls on the audio track. 
The multimodal approach aims to enhance the system's robustness and reliability. A late fusion approach is employed, utilising two independently trained subnetworks for each modality and a fusion network that merges the outputs of the two subnetworks. A YOLOv7-tiny model trained on images of birds of prey is employed for the image modality, and a light-VGG11 model trained on Mel-spectrograms of chicken alarm calls is used as the audio subnetwork. 

## YOLOv7-tiny Image Subnetwork
is adopted without any changes from another work. The model used for the fusion network, can be found in the Fusion folder named "yolov7-best.pt"

## light-VGG11 Audio Subnetwork
can be found in the folder light-VGG11. Most classes are sourced from here: https://github.com/Max-1234-hub/light-VGG11

## Fusion Network
can be found in the folder Fusion 