# [One-stage Deep Edge Detection Based on Dense-scale Feature Fusion and Pixel-level Imbalance Learning](https://ieeexplore.ieee.org/document/9960785)

## Requirements

pytorch >= 1.0

torchvision

## Training & Testing

### Data preparation

-Download the [BSDS500]() and the [NYUDv2](http://vcl.ucsd.edu/hed/nyu/)
-Place the images to "./data/.'

####
- For NYUDv2 dataset, the following command can bu run for data augmentation
```
python ./data/aug.py
```

### Pretrained Models
- Download the pretrained model "./pretrained/bsds"

### Training and Testing
- Download the pre-trained model vgg16-bn and EfficientNetv2-S
- model.py represents the use of EfficientNetv2-s model
- model_vgg.py represents the use of VGG16-bn model
```
python main.py
```

### Eval
- The evaluation codes are provided in "./eval"(This is the matlab code)
