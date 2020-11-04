import torch
from torch import nn
from torch import optim
# from model import VQAModule
from vqa.model import VQANoImageModule
from torchvision import transforms
from torch.utils.data import dataloader
from vqa.vqa_dataset import VQADataset, collate_fn
import tqdm
import numpy as np
import utlis
# from vqa_dataset import VQADataset, collate_fn

USE_GPU = True
CUDA_INDEX = 1
device = torch.device(f'cuda:{CUDA_INDEX}') if USE_GPU else torch.device('cpu')

batch_size = 128
embedding_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 1e-3
epoch_size = 2

train_image_root_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/train'
val_image_root_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/train'
train_question_json_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
val_question_json_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
model_save_path = '/home/ubuntu/likun/image_save_kernels/vqa_clevr_noimage.pt'

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
train_num = 50000
train_data = VQADataset(train_image_root_path, train_question_json_path, transform=transform, sample_num=train_num)
train_loader = dataloader.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_num = 5000
val_data = VQADataset(val_image_root_path, val_question_json_path, transform=transform, sample_num=val_num)
val_loader = dataloader.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

q_vocab_size = len(train_data.question_field.vocab.itos)
a_vocab_size = len(train_data.answer_field.vocab.itos)
model = VQANoImageModule(q_vocab_size, a_vocab_size, embedding_size, hidden_size, num_layers, output_size=hidden_size).to(device)
criterion = nn.CrossEntropyLoss()

params = [param for name, param in model.named_parameters() if 'resnet' not in name]
optimizer = optim.Adam(params=params, lr=learning_rate)

train_config = {
    'task_name': 'image_caption',
    'train_data_info': train_data.dataset_info,
    'batch_size': batch_size,
    'embedding_size': embedding_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'learning_rate': learning_rate,
    'epoch_size': epoch_size,
    'q_vocab_size': q_vocab_size,
    'a_vocab_size': q_vocab_size,
    'img_root_path': train_image_root_path,
    'caption_path': train_question_json_path
}

for epoch in range(1, epoch_size + 1):
    losses = []
    model.train()
    train_batch_num = 0
    for batch in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}, Training'):
        optimizer.zero_grad()
        img = batch[0].to(device)
        questions = batch[1].to(device)
        lengths = batch[2]
        answers = batch[3].to(device)
        output = model.forward(img, questions, lengths)
        loss = criterion(output, answers)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        train_batch_num += 1
        if train_batch_num % 20 == 0:
            print(f'Epoch {epoch}, Batch {train_batch_num}, Mean loss: {np.mean(losses)}')
    print(f'Epoch {epoch}, Mean loss: {np.mean(losses)}')
    checkpoint = {
        'net': model.state_dict(),
        'q_vocab': train_data.question_field.vocab,
        'a_vocab': train_data.answer_field.vocab,
        'img_transform': transform,
        'config': train_config
    }
    torch.save(checkpoint, model_save_path)

    losses = []
    model.eval()
    true_labels = []
    predict_labels = []
    for batch in tqdm.tqdm(val_loader, desc=f'Epoch {epoch}, validating'):
        optimizer.zero_grad()
        img = batch[0].to(device)
        questions = batch[1].to(device)
        lengths = batch[2]
        answers = batch[3].to(device)
        output = model.forward(img, questions, lengths)
        loss = criterion(output, answers)
        losses.append(loss.item())
        _, predict = torch.max(output.data, 1)
        true_labels.extend(answers.tolist())
        predict_labels.extend(predict.tolist())
    acc, p, r, f1 = utlis.evaluate(true_labels, predict_labels)
    print(f'Epoch {epoch}, Mean loss: {np.mean(losses)}, ACC: {acc}, p:{p}, r:{r}, f:{f1}')

