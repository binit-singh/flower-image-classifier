# Stdlib imports
import json
import torch
from torch import optim
from torchvision import (
    transforms,
)
from PIL import Image

# Local imports
from models import arch_model


def load_checkpoint(checkpoint_path):
    """
    It will load checkpoint from given location and return model.

    Args:
        checkpoint_path (str): Location where checkpoint was saved

    Returns:
        model (torchvision.models)
    """
    checkpoint = torch.load(checkpoint_path)
    model = arch_model[checkpoint['arch']]
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.classifier = checkpoint['classifier']
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model

    Args:
        image (list)

    Returns:
        transformed image
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(Image.open(image))


def label_mapping(file_name):
    """
    It will create a mapping from category label to category name.

    Args:
        file_name (str): Filename of category to name.

    Returns:
        cat_to_name (dict)
        {
            '21': 'fire lily',
            '3': 'canterbury bells',
            '45': 'bolero deep blue'
        }
    """
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
