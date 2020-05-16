# Lib imports
from torchvision import models

densenet = models.densenet121(pretrained=True)
vgg = models.vgg13(pretrained=True)
alexnet = models.alexnet(pretrained=True)

arch_model = {'densenet': densenet, 'vgg': vgg, 'alexnet': alexnet}
arch_input = {'densenet': 1024, 'vgg': 25088, 'alexnet': 9216}
