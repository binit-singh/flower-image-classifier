import torch

# Local imports
from get_input_args import get_predict_args
from predict_helper import load_checkpoint, process_image, get_device, label_mapping


def predict():
    """
    This function will predict the image class.
    """
    args = get_predict_args()
    model = load_checkpoint(args.checkpoint_path)
    device = get_device(args.gpu)

    model.eval()
    model.to(device)

    invert_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    cat_to_name = label_mapping(args.category_names)

    image = process_image(args.image_path)
    image = image.unsqueeze_(0)
    image = image.to(device)

    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(args.top_k)

    top_class = [invert_class_to_idx[idx.item()] for idx in top_idx[0].data]
    probs = [p.item() for p in top_p[0].data]
    names = [cat_to_name[idx] for idx in top_class]

    print('\n\n************ Result *************\n')
    for name, probability in zip(names, probs):
        print(f"Name: {name} \nProbability: {probability * 100:.2f} %\n\n")
    print('\n**********************************\n\n')


if __name__ == '__main__':
    predict()