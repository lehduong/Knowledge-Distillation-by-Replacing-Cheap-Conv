# Plug (Cheap Conv) and Play
This repository contains the code for Knowledge Distillation in Convolutional Neural Network in **Low-resource settings** (both computational cost and data). The code is usable for both Classification and Semantic Segmantation tasks. 
## Usage
Follow below instructions for running knowledge distillation in DeepWV3Plus model for Cityscapes dataset:

1. Download the dataset to `data` folder, the folder structure should be: `data\gtFine\train` as the sample set provided

2. Download the pretrained model as a teacher from [here](https://github.com/NVIDIA/semantic-segmentation) and put it in `checkpoints`

3. Try `python train.py -c config.json` to run code with default settings.

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

## Results

To be updated ....

## Acknowledgements
This repository is borrowed from the project [Pytorch-Template](https://github.com/victoresque/pytorch-template) and [NVIDIA semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)
