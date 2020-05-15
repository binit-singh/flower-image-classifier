import json
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

densenet = models.densenet121(pretrained=True)
vgg = models.vgg13(pretrained=True)
alexnet = models.alexnet(pretrained=True)

arch_model = {'densenet': densenet, 'vgg': vgg, 'alexnet': alexnet}
arch_input = {'densenet': 1024, 'vgg': 25088, 'alexnet': 9216}


def prepare_dataset(data_dir, batch_size=32):
    """
    It will prepare train, valid and test dataset.

    Args:
        data_dir (str): It's path of data directory.
        batch_size (int): Number if image in one batch.

    Return
        image_dataset (dict): It will contain training, validation and test dataset and dataloaders
        image_dataset = {
            'train_dataset': [...],
            'train': [...],
            'valid': [...],
            'valid_dataset': [...],
            'test': [...],
            'test_dataset': [...]
        }
    """
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(244),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])
    ])
    testing_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size)

    data = {
        'train_dataset': train_datasets,
        'train': train_dataloaders,
        'valid_dataset': validation_dataset,
        'valid': validation_dataloader,
        'test_dataset': testing_dataset,
        'test': testing_dataloader
    }

    return data


def get_device(user_request):
    """
    It will return processor type i.e GPU/CPU depending upon user's request and availability.

    Args:
        user_request (str): gpu/cpu

    Returns:
        cuda device
    """
    if user_request == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return torch.device(device)


def get_model(arch, hidden_input, processor_type):
    """
    It will return pre trained model.

    It allows users to choose from at least two different architectures available from torchvision.models

    Args:
        arch (str): user option to chosse from available options
        hidden_input (int): Number of nodes in hidden layer.
        processor_type (str): User's choise to use GPU/CPU

    Returns:
        model (torchvision.models)
    """
    model = arch_model[arch]

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(arch_input[arch], hidden_input),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_input, 102),
        nn.LogSoftmax(dim=1)
    )
    device = get_device(processor_type)
    model.to(device)

    return model


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


def save_checkpoint(model, optimizer, dataset, epochs, arch, learning_rate, checkpoint_path):
    """
    Save the checkpoint.

    Args:
        model (torchvision.models): Trained Model.
        optimizer (torch.optim): Optimizer
        dataset (list): List of training images.
        epochs (int): Number of epochs
        arch (str): Type of architecture user have choosen
        learning_rate (float): Rate at which model was learning.
        checkpoint_path (str): File location where checkpoint should be saved.

    Returns:
        None
    """
    input_feature = arch_input[arch]
    checkpoint = {
        'input_size': input_feature,
        'output_size': 102,
        'arch': arch,
        'classifier': model.classifier,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': dataset.class_to_idx
    }
    torch.save(checkpoint, checkpoint_path)