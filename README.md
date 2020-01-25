# On searching an effective and efficient Pipeline for Distillating Knowledge in Convolutional Neural Networks
This repository contains the code for Knowledge Distillation in Convolutional Neural Network in **Low-resource settings**. The code is usable for both Classification and Semantic Segmantation tasks. 
## Usage
Following below instructions for running with Cityscapes dataset:

Download the dataset to `data` folder, the folder structure should be: `data\gtFine\train` as the sample set provided

Download the pretrained model as a teacher from [here](https://github.com/NVIDIA/semantic-segmentation) and put it in `checkpoints`

Try `python train.py -c config.json` to run code with default settings.

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

## Acknowledgements
This repository is borrowed from the project [Pytorch-Template](https://github.com/victoresque/pytorch-template) and [NVIDIA semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)
