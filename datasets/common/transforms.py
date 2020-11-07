import torchvision

def get_image_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
    return transform

