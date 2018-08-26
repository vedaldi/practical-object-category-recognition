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
    def __init__(self, class_name, set_name):
        self.imdb = lab.get_image_database()
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        c = self.imdb['class_names'].index(class_name)
        s = self.imdb['set_names'].index(set_name)
        self.indices = [i for i, (c_,s_) in enumerate(
            zip(self.imdb['classes'], self.imdb['sets'])) if c_==c and s_==s]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image_name = f"{self.imdb['images'][self.indices[index]]}"
        image_path = os.path.join('data', 'images', f"{image_name}.jpg")
        image = Image.open(image_path)
        image = self.normalize(image)
        return {'image': image, 'name': image_name}

# Get the encoder CNN
model = lab.get_encoder_cnn()

# Encode all images
names = []
codes = [[] for _ in range(7)]
imdb = lab.get_image_database()
for class_name in imdb['class_names']:
    for set_name in imdb['set_names']:
        print(f"{class_name} for {set_name}")

        # Get a data loader
        loader = torch.utils.data.DataLoader(
            AllImages(class_name, set_name),
            batch_size=64,
            num_workers=0,
            shuffle=False
        )

        # Process image set
        for batch_index, batch in enumerate(loader):
            print(f"Processing {batch_index+1} of {len(loader)}")
            names.extend(batch['name'])
            with torch.no_grad():
                codes_ = model(batch['image'])    
            for i in range(7):
                codes[i].append(codes_[i])

        # Compact code tensors
        for i in range(7):
            codes[i] = [torch.cat(codes[i], 0)]

# Find correspondences between codes and imdb
indices = []
for name in imdb['images']:
    index = names.index(name)
    indices.append(index)

# Save everything back
for i in range(7):
    codes[i] = codes[i][0][indices,:]
    path = os.path.join("data", f"conv{i+1}.pth")    
    torch.save({'codes': codes[i]}, path)
    print(f"Saved {path} {list(codes[i].shape)}")



