from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment
from PIL import Image

def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def _convert_to_rgb(image):
    return image.convert('RGB')

def clip_train_transform(size=800):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(size,scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    
    return train_transform

def clip_test_transform(size=800):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
        ])
    
    return test_transform