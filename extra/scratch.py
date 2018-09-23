# Scratch file

import os
import lab
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.utils.data
import torchvision

imdb = lab.get_image_database()
sel = lab.get_indices(imdb, 'aeroplane')

if False:
    # Test encoder soundness
    im = Image.open('../practical-cnn/data/peppers.png')
    model = lab.get_encoder_cnn()
    codes = model(model.normalize(im)[None,:])
    import json
    with open('../practical-cnn/data/imnet_classes.json') as f:
        classes = json.load(f)
    bestk = torch.argmax(codes[7])
    name = classes[str(bestk.item())][1]
    print(name)

imdb = lab.get_image_database()

# Get the data vectors and normalize them
x = lab.get_codes(layer=7)

if False:
    model = lab.get_encoder_cnn()
    for t in range(0,len(imdb['images']),17):
        image_name = f"{imdb['images'][t]}"
        image_path = os.path.join('data', 'images', f"{image_name}.jpg")
        image = Image.open(image_path)
        image = model.normalize(image)
        codes = model(image[None,:])
        a = codes[7-1].reshape(-1)
        b = x[t]
        print(image_name, ((a-b)**2).mean().item())

x = torch.sqrt(x)
x /= torch.sqrt((x * x).sum(1))[:,None]

# Get the training data and its labels
pos = lab.get_indices(imdb, 'motorbike', 'train')
neg = lab.get_indices(imdb, ['aeroplane', 'person', 'background'], 'train')

pos = pos[:200]
neg = neg[:200]
x_train = x[torch.cat((pos,neg))]
c_train = torch.tensor([1] * len(pos) + [-1] * len(neg), dtype=torch.float32)

plt.figure(10)
lab.imsc(x_train[None,:])

# Train the SVM
#w, b = lab.svm_sgd(x_train, c_train, lam=0.0001, num_epochs=120)
w, b = lab.svm_sdca(x_train, c_train, lam=1e-4, num_epochs=600)

# Classify
scores_train = x_train @ w + b

plt.figure(2)
plt.clf()
_, _, ap_train = lab.pr(c_train, scores_train)

# Get the valing data and its labels
pos = lab.get_indices(imdb, 'motorbike',  'val')
neg = lab.get_indices(imdb, ['aeroplane', 'person', 'background'], 'val')
x_val = x[torch.cat((pos, neg))]
c_val = torch.tensor([1] * len(pos) + [-1] * len(neg), dtype=torch.float32)
scores_val = x_val @ w + b

_, _, ap_val = lab.pr(c_val, scores_val)

plt.legend((f'train ({100*ap_train:.1f}%)', f'val ({100*ap_val:.1f}%)'))
plt.pause(0)
