from PIL import Image
import torch
from torchvision import transforms
from model import ImageCaptionModule
import os
import random
import shutil

USE_GPU = True
CUDA_INDEX = 1
device = torch.device(f'cuda:{CUDA_INDEX}') if USE_GPU else torch.device('cpu')

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

def get_image(image_path, img_transform=None):
    image = Image.open(image_path).convert('RGB')
    if image.size[0] < 225 or image.size[1] < 255:
        image = image.resize((256, 256), Image.ANTIALIAS)
    if img_transform is not None:
        image_tensor = img_transform(image)
    else:
        image_tensor = transforms.ToTensor(image)
    return image_tensor, image


def load_model(model_save_path):
    checkpoint = torch.load(model_save_path)
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    img_transform = checkpoint['img_transform']
    model = ImageCaptionModule(config['embedding_size'], config['hidden_size'],
                               config['num_layers'], config['vocab_size'])
    model.load_state_dict(checkpoint['net'])
    return model, img_transform, vocab

def predict_image_caption(image_path, model, img_transform, vocab, max_length=20):
    image_tensor, image = get_image(image_path, img_transform)

    image_tensor = image_tensor.unsqueeze(0).to(device)
    sampled_ids = model.sample(image_tensor, max_length=max_length, eos_index=vocab.stoi['<eos>'])
    for ids in sampled_ids:
        words = [vocab.itos[x] for x in ids.tolist()]
        print(' '.join(words))

def main():
    # Build model
    model_save_path = '/home/ubuntu/likun/image_save_kernels/image_caption_flickr8k_cnnrnn_1014.pt'
    model, img_transform, vocab = load_model(model_save_path)
    model.to(device)
    model.eval()

    # Read and predict image caption
    image_root_path = '/home/ubuntu/likun/image_data/flickr30k-images'
    image_names = os.listdir(image_root_path)
    image_name = random.choice(image_names)
    image_path = os.path.join(image_root_path, image_name)
    predict_image_caption(image_path, model, img_transform, vocab)

    # Save and show the image
    synchronization_dir = '/home/ubuntu/likun/windows_temp_data'
    tempfiles = os.listdir(synchronization_dir)
    for f in tempfiles:
        os.remove(os.path.join(synchronization_dir, f))
    shutil.copy(image_path, os.path.join(synchronization_dir, image_name))

if __name__ == '__main__':
    main()

