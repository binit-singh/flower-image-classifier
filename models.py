# Lib imports
from torchvision import models

dense_net = models.densenet121(pretrained=True)
vgg = models.vgg13(pretrained=True)
alex_net = models.alexnet(pretrained=True)

arch_model = {'densenet': dense_net, 'vgg': vgg, 'alexnet': alex_net}
arch_input = {'densenet': 1024, 'vgg': 25088, 'alexnet': 9216}
