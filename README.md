# Image-classifier
This is an image classifier built with PyTorch. I have trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. 
The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

## Usage
The project includes two imprtant files `train.py` and `predict.py`. The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image.

### Train a network - `python train.py data_directory`

* It Prints out training loss, validation loss, and validation accuracy as the network trains
* You can Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
* Choose architecture: `python train.py data_dir --arch "densenet"`
* Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: `python train.py data_dir --gpu`

### Predict Image - `python predict.py /path/to/image /path/to/checkpoint`

* It Predict flower name from an image along with the probability of that name.
