# Import python module
import argparse


def get_train_args():
    """
    Retrive and parse the command line arguments provided by user when they train the model from terminal.
    
    Command Line arguments
    1. Data directory as --dir with default value /flowers
    2. Save directory as --save_dir with default value /checkpoint
    3. Model Architecture as --arch with default value densenet
    4. Learning rate as --learning_rate with default as 0.001
    5. Hidden units as --hidden_units with default 512
    6. Epochs as --epoch with default 3
    7. GPU as --gpu
    """
    
    parser = argparse.ArgumentParser(
        prog='train.py',
        usage='python %(prog)s data_directory',
        description='It will train a new network on a dataset and save the model as a checkpoint.'
    )
    
    parser.add_argument(
        'dir',
        type=str,
        help='Data directory to train model'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoint/checkpoint.pth',
        help='Data directory to Save checkpoint.'
    )
    parser.add_argument(
        '--arch',
        type=str,
        default='densenet',
        help='Model architecture',
        choices=['densenet', 'vgg', 'alexnet']
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=512,
        help='Hidden units'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=5, 
        help='Epochs'
    )
    parser.add_argument(
        '--gpu',
        const='gpu',
        nargs='?',
        default='cpu'
    )
    
    return parser.parse_args()


def get_predict_args():
    """
    Retrive and parse the command line arguments provided by user when they predict result from terminal.
    
    Command line arguments
    1. Image path as --image_path with default value flowers/test/1/image_06743.jpg
    2. Checkpoint path as --checkpoint_path with default value checkpoint/checkpoint.pth
    3. Number of Top class as --top_k with default value 3
    4. Category names as --category_names with default value cat_to_name.json
    5. GPU as --gpu
    """
    parser = argparse.ArgumentParser(
        prog='predict.py',
        usage='python %(prog)s /path/to/image checkpoint',
        description="""
        Predict flower name from an image with predict.py along with the probability of that name.
         That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
         """
        
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path of image you want to predict.'
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path of checkpoint',
        default='checkpoint.pth'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        help='Return top K most likely classes',
        default=1
    )
    parser.add_argument(
        '--category_names',
        type=str,
        help='Use a mapping of categories to real names.',
        default='cat_to_name.json'
    )
    parser.add_argument(
        '--gpu',
        const='gpu',
        nargs='?',
        default='cpu',
        help='Use GPU for inference.'
    )
    
    return parser.parse_args()
