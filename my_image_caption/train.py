import torch
from torch import nn
from torch import optim
from model import ImageCaptionModule
from torchvision import transforms
from torch.utils.data import dataloader
from image_caption_dataset import ImageCaptionDataset, collate_fn
import tqdm
import numpy as np

USE_GPU = True
CUDA_INDEX = 1
device = torch.device(f'cuda:{CUDA_INDEX}') if USE_GPU else torch.device('cpu')

batch_size = 128
embedding_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 1e-3
epoch_size = 5

root_path = '/home/ubuntu/likun/image_data/flickr8k-images'
caption_path = '/home/ubuntu/likun/image_data/caption/dataset_flickr8k.json'
model_save_path = '/home/ubuntu/likun/image_save_kernels/image_caption_flickr8k_cnnrnn_1014.pt'

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

train_data = ImageCaptionDataset(root_path, caption_path, transform=transform)
train_loader = dataloader.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
vocab_size = len(train_data.caption_field.vocab.itos)
model = ImageCaptionModule(embedding_size, hidden_size, num_layers, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()

params = [param for name, param in model.named_parameters() if 'encoder.resnet' not in name]
optimizer = optim.Adam(params=params, lr=learning_rate)

train_config = {
    'task_name': 'image_caption',
    'train_data': train_data.dataset_name,
    'batch_size': batch_size,
    'embedding_size': embedding_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'learning_rate': learning_rate,
    'epoch_size': epoch_size,
    'vocab_size': vocab_size,
    'img_root_path': root_path,
    'caption_path': caption_path
}

for epoch in range(1, epoch_size + 1):
    losses = []
    model.train()
    train_batch_num = 0
    for batch in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}, Training'):
        optimizer.zero_grad()
        img = batch[0].to(device)
        caption = batch[1].to(device)
        lengths = batch[2]
        target = nn.utils.rnn.pack_padded_sequence(caption, lengths, batch_first=True)[0]
        output = model.forward(img, caption, lengths)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        train_batch_num += 1
        if train_batch_num % 20 == 0:
            print(f'Epoch {epoch}, Batch {train_batch_num}, Mean loss: {np.mean(losses)}')
    print(f'Epoch {epoch}, Mean loss: {np.mean(losses)}')
    checkpoint = {
        'net': model.state_dict(),
        'vocab': train_data.caption_field.vocab,
        'img_transform': transform,
        'config': train_config
    }
    torch.save(checkpoint, model_save_path)
