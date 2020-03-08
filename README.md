# Plug (Cheap Conv) and Play
This repository contains the code for Knowledge Distillation in Convolutional Neural Network in **Low-resource settings** (both computational cost and data). The code is usable for both Classification and Semantic Segmantation tasks. 
## Usage
Follow below instructions for running knowledge distillation in DeepWV3Plus model for Cityscapes dataset:

1. Download the dataset to `data` folder, the folder structure should be: `data\gtFine\train` as the sample set provided

2. Download the pretrained model as a teacher from [here](https://github.com/NVIDIA/semantic-segmentation) and put it in `checkpoints`

3. Try `python train.py -c config.json` to run code with default settings.

## Supports

### Resuming from checkpoints
You can resume from a previously saved checkpoint by adding `resume_path` in `trainer` of `config.json` such as:
```
"metrics": [],
"trainer": {
        ...
        "do_validation_interval": 100,
        "len_epoch": 100,
        "resume_path": "checkpoint-epoch20.pth",
        "tensorboard": true
    },
"lr_scheduler": {
...
```

## Results

In our experiments, the student networks are finetuned with **unlabeled** images and usually requires **less than 2 hours** (on single P100 GPU) to achieve the results below. 

### Cityscapes

Results on **Test** set. Note that all the submission are augmented with *sliding* windows and crop size of 1024.

|  Model | Teacher Param | Student Param | Teacher mIoU | Student mIoU |
| ------ | ------------- | ------------- | ------------ | ------------ |
| Deeplabv3+ (Wideresnet38) Video augmented |  137M | 92M | 83.4 | 82.1 |
| Deeplabv3+ (Wideresnet38) Video augmented |  137M | 79M | 83.4 | 81.3 |

Results on **Val** set. We **don't** use test-time augmentation on val set.
|  Model | Teacher Param | Student Param | Teacher mIoU | Student mIoU |
| ------ | ------------- | ------------- | ------------ | ------------ |
| Gated-SCNN |  137M | 86M | 80.9 | 79.6 |

## Acknowledgements
This repository is borrowed from the project [Pytorch-Template](https://github.com/victoresque/pytorch-template) and [NVIDIA semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)
