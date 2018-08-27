import os
import lab
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.utils.data
import torchvision

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

class AllImages(torch.utils.data.Dataset):
    def __init__(self):
        self.imdb = lab.get_image_database()
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self.normalize_flip = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.RandomHorizontalFlip(1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.imdb['images'])

    def __getitem__(self, index):
        image_name = f"{self.imdb['images'][index]}"
        jitter = self.imdb['jitters'][index]
        image_path = os.path.join('data', 'images', f"{image_name}.jpg")
        print(f"Processing image {image_name} flipped {jitter}")
        image = Image.open(image_path)
        if jitter == 1:
            image = self.normalize_flip(image)
        else:
            image = self.normalize(image)
        return {'image': image, 'name': image_name}

# Get the encoder CNN
model = lab.get_encoder_cnn()

# Get a data loader
loader = torch.utils.data.DataLoader(
    AllImages(),
    batch_size=64,
    num_workers=0,
    shuffle=False
)

# Process image set
names = []
codes = [[] for _ in range(7)]
for batch_index, batch in enumerate(loader):
    print(f"Processing {batch_index+1} of {len(loader)}")
    names.extend(batch['name'])
    with torch.no_grad():
        codes_ = model(batch['image'])
    for i in range(7):
        codes[i].append(codes_[i])

    # Compact code tensors
    if (batch_index % 10 == 0) or (batch_index + 1 == len(loader)):
        for i in range(7):
            codes[i] = [torch.cat(codes[i], 0)]

# Save everything back
for i in range(7):
    codes[i] = codes[i][0]
    path = os.path.join("data", f"conv{i+1}.pth")    
    torch.save({'codes': codes[i]}, path)
    print(f"Saved {path} {list(codes[i].shape)}")



