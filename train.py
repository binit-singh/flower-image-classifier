import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
from PIL import Image

# Local imports
from get_input_args import get_train_args
from train_helper import prepare_dataset, get_model, save_checkpoint, get_device
from workspace_utils import active_session


def train():
    """
    It successfully trains a new network on a dataset of images.
    """
    args = get_train_args()
    data = prepare_dataset(args.dir)
    device = get_device(args.gpu)
    model = get_model(args.arch, args.hidden_units, args.gpu)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    steps = 0
    running_loss = 0
    print_every = 32

    with active_session():
        for epoch in range(args.epoch):

            # Train model
            for images, labels in tqdm(data['train'], desc='Train'):
                steps += 1
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Validate model
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in data['valid']:
                            images, labels = images.to(device), labels.to(device)
                            logps = model(images)
                            loss = criterion(logps, labels)
                            test_loss += loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print('***' * 10)
                    print(f"Epoch {epoch + 1}/{args.epoch}.. ")
                    print(f"Train loss: {running_loss / print_every:.3f}.. ")
                    print(f"Test loss: {test_loss / len(data['valid']):.3f}.. ")
                    print(f"Accuracy: {(accuracy / len(data['valid'])) * 100:.2f} %")
                    print('***' * 10)
                    running_loss = 0
                    model.train()

    # Save the checkpoint
    save_checkpoint(
        model,
        optimizer,
        data['train_dataset'],
        args.epoch,
        args.arch,
        args.learning_rate,
        args.save_dir
    )


if __name__ == '__main__':
    train()
