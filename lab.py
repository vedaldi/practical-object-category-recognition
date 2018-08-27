import os
import io
import math
import random
import requests
import time
import torch
import torch.nn as nn
import torch.nn.functional  as F
import torchvision
import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt

def t2im(x):
    """Rearrange the N x K x H x W to have shape (NK) x 1 x H x W.
    
    Arguments:
        x {torch.Tensor} -- A N x K x H x W tensor.
    
    Returns:
        torch.Tensor -- A (NK) x 1 x H x W tensor.
    """
    return x.reshape(-1, *x.shape[2:])[:,None,:,:]

def imread(file):
    """Read the image `file` as a PyTorch tensor.
    
    Arguments:
        file {str} -- The path to the image.
    
    Returns:
        torch.Tensor -- The image read as a 3 x H x W tensor in the [0, 1] range.
    """
    # Read an example image as a NumPy array
    x = Image.open(file)
    x = np.array(x)
    if len(x.shape) == 2:
        x = x[:,:,None]
    return torch.tensor(x, dtype=torch.float32).permute(2,0,1)[None,:]/255

def imsc(im, *args, quiet=False, **kwargs):
    """Rescale and plot an image represented as a PyTorch tensor.
    
     The function scales the input tensor im to the [0 ,1] range.
    
    Arguments:
        im {torch.Tensor} -- A 3 x H x W or 1 x H x W tensor.
    
    Keyword Arguments:
        quiet {bool} -- Do not plot. (default: {False})
    
    Returns:
        torch.Tensor -- The rescaled image tensor.
    """
    handle = None
    with torch.no_grad():
        im = im - im.min() # make a copy
        im.mul_(1/im.max())
        if not quiet:
            bitmap = im.expand(3, *im.shape[1:]).permute(1,2,0).numpy()
            handle = plt.imshow(bitmap, *args, **kwargs)
            ax = plt.gca()
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    return im, handle

def imarraysc(tiles, spacing=0, quiet=False):
    """Plot the PyTorch tensor `tiles` with dimesion N x C x H x W as a C x (MH) x (NW) mosaic.    
    
    The range of each image is individually scaled to the range [0, 1].

    Arguments:
        tiles {[type]} -- [description]
    
    Keyword Arguments:
        spacing {int} -- Thickness of the border (infilled with zeros) around each tile (default: {0})
        quiet {bool} -- Do not plot the mosaic. (default: {False})
    
    Returns:
        torch.Tensor -- The mosaic as a PyTorch tensor. 
    """
    handle = None
    num = tiles.shape[0]
    num_cols = math.ceil(math.sqrt(num))
    num_rows = (num + num_cols - 1) // num_cols
    c = tiles.shape[1]
    h = tiles.shape[2]
    w = tiles.shape[3]
    mosaic = torch.zeros(c,
      h*num_rows + spacing*(num_rows-1),
      w*num_cols + spacing*(num_cols-1))
    for t in range(num):
        u = t % num_cols
        v = t // num_cols
        tile = tiles[t]
        mosaic[0:c,
          v*(h+spacing) : v*(h+spacing)+h,
          u*(w+spacing) : u*(w+spacing)+w] = imsc(tiles[t], quiet=True)[0]
    return imsc(mosaic, quiet=quiet)
    
def get_image_database():
    "Get the image database `imdb`."
    class_names = ["aeroplane", "motorbike", "person", "background", "horse", "car"]
    set_names = ["train", "val"]
    jitter_names = ["normal", "flipped"]

    # Read a text file
    def read_text(path):
        with open(path) as f:
            return [name.strip() for name in f.readlines()]

    # Source all the images
    imdb = dict()
    imdb['images'] = read_text(os.path.join('data', 'all.txt')) * 2
    n = len(imdb['images'])
    imdb['jitters']      = torch.tensor([0] * (n//2) + [1] * (n//2), dtype=torch.int8)
    imdb['sets']         = torch.zeros(n).byte() - 1
    imdb['is_aeroplane'] = torch.zeros(n).byte()
    imdb['is_motorbike'] = torch.zeros(n).byte()
    imdb['is_person']    = torch.zeros(n).byte()
    imdb['is_horse']     = torch.zeros(n).byte()
    imdb['is_car']       = torch.zeros(n).byte()
    imdb['is_background']= torch.zeros(n).byte()
    imdb['class_names']  = class_names
    imdb['set_names']    = set_names
    imdb['jitter_names'] = jitter_names

    # Index images
    index = dict()
    for i, name in enumerate(imdb['images']):
        if name in index:
            index[name] += (i,)
        else:
            index[name] = (i,)

    # Fill metadata
    for set_index, set_name in enumerate(set_names):        
        for class_index, class_name in enumerate(class_names):
            content = read_text(os.path.join("data", f"{class_name}_{set_name}.txt"))
            for name in content:
                for i in index[name]:
                    imdb['sets'][i] = set_index
                    imdb[f'is_{class_name}'][i] = 1

    return imdb

def get_indices(imdb, classes=None, sets=None, jitters=0, minus=None):
    """Find the indices of the images satisfying the specified criteria.

    The function returns the intersection of the specified criteria. Each criterion
    can be specified as a list of values, which are or-ed.
    
    Arguments:
        imdb {dict} -- The `imdb` database structure.
    
    Keyword Arguments:
        classes {None, str, int, list} -- A list of class names or ids to select (default: {None})
        sets {None, str, int, list} -- A list of set names or ids to select (default: {None})
        jitters {None, str, int, list} -- A list of jitter names or ids to select (default: {0})
        minus {torch.Tensor} -- Indices to exclude (default: {None})
    
    Returns:
        [type] -- [description]
    """
    with torch.no_grad():
        if not type(classes) in [list, tuple]:
            classes = [classes]
        if not type(sets) in [list, tuple]:
            sets = [sets]
        if not type(jitters) in [list, tuple]:
            jitters = [jitters]
        n = len(imdb['images'])
        sel1 = torch.zeros(n).byte()
        for c in classes:
            if c is not None:            
                if not isinstance(c, str):
                    c = imdb['class_names'][c]
                sel1 |= imdb[f'is_{c}']
        sel2 = torch.zeros(n).byte()
        for s in sets:
            if s is not None:
                if isinstance(s, str):
                    s = imdb['set_names'].index(s)
                sel2 |= (imdb['sets'] == s)
        sel3 = torch.zeros(n).byte()
        for j in jitters:
            if j is not None:
                if isinstance(j, str):
                    j = imdb['jitter_names'].index(j)
                sel3 |= (imdb['jitters'] == j)        
        selection = sel1 & sel2 & sel3
        if minus is not None:
            selection[minus] = 0
    return torch.nonzero(selection).reshape(-1)

class Encoder(nn.ModuleDict):
    "A wrapper around AlexNet to encode images."
    def __init__(self, alexnet):
        super(nn.ModuleDict, self).__init__()
        layers = list(alexnet.features.children()) + list(alexnet.classifier.children())
        self['conv1'] = layers[0]
        self['relu1'] = layers[1]
        self['pool1'] = layers[2]
        self['conv2'] = layers[3]
        self['relu2'] = layers[4]
        self['pool2'] = layers[5]
        self['conv3'] = layers[6]
        self['relu3'] = layers[7]
        self['conv4'] = layers[8]
        self['relu4'] = layers[9]
        self['conv5'] = layers[10]
        self['relu5'] = layers[11]
        self['pool5'] = layers[12]
        self['fc6']   = layers[14]
        self['relu6'] = layers[15]
        self['fc7']   = layers[17]
        self['relu7'] = layers[18]
        self['fc8']   = layers[19]
        self['avg']   = nn.AdaptiveAvgPool2d((2,2),)
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def encode(im):
        """Encode an image.
        
        Arguments:
            im {PIL.Image} -- Image in PIL format.
        
        Returns:
            [torch.Tensor] -- A list of tensors, one for each code vector.
        """
        x = self.normalize(im)
        with torch.no_grads():
            return self(x)

    def forward(self, x):
        """Forwad evaluation.

        Evaluate the encoder forward.
         
        Arguments:
           x {torch.Tensor} -- Image batch as a PyTorch tensor.
        
        Returns:
            [torch.Tensor] -- A list of tensors, one for each code vector.
        """
        n = len(x)
        output = ()
        x = self['conv1'](x)
        x = self['relu1'](x)
        x = self['pool1'](x)
        output += (self['avg'](x).reshape(n,-1),)
        x = self['conv2'](x)
        x = self['relu2'](x)
        x = self['pool2'](x)
        output += (self['avg'](x).reshape(n,-1),)
        x = self['conv3'](x)
        x = self['relu3'](x)
        output += (self['avg'](x).reshape(n,-1),)
        x = self['conv4'](x)
        x = self['relu4'](x)
        output += (self['avg'](x).reshape(n,-1),)
        x = self['conv5'](x)
        x = self['relu5'](x)
        x = self['pool5'](x)
        output += (self['avg'](x).reshape(n,-1),)
        x = x.reshape(n,-1)
        x = self['fc6'](x)
        x = self['relu6'](x)
        output += (x.reshape(n,-1),)
        x = self['fc7'](x)
        x = self['relu7'](x)
        output += (x.reshape(n,-1),)
        x = self['fc8'](x)
        output += (x.reshape(n,-1),)
        return output

def get_encoder_cnn():
    "Get the encoder neural network model."
    model_path = os.path.join('data', 'alexnet.pth')
    if not os.path.exists(model_path):
        alexnet = torchvision.models.alexnet(pretrained=True)
        torch.save(alexnet.state_dict(), model_path)
    alexnet = torchvision.models.alexnet(pretrained=False)
    alexnet.load_state_dict(torch.load(model_path))
    model = Encoder(alexnet) 
    return model

def get_codes(layer=7):
    "Get the codes for an image in `imdb` and the specified `layer`."
    data = torch.load(os.path.join('data', f"conv{layer}.pth"))
    return data['codes']

def get_image(imdb, index):
    "Get an image in `imdb` as a PyTorch tensor."
    image_path = os.path.join('data', 'images', f"{imdb['images'][index]}.jpg")
    return imread(image_path)

def get_pil_image(imdb, index):
    "Get an image in `imdb` in PIL format."
    image_path = os.path.join('data', 'images', f"{imdb['images'][index]}.jpg")
    return Image.open(image_path)

def svm_sgd(x, c, lam=0.01, learning_rate=0.01, num_epochs=30):
    "Train an SVM using the SGD algorithm."
    with torch.no_grad():
        xb = 1
        n = x.shape[0]
        d = x.shape[1]
        w = torch.randn(d)
        b = torch.zeros(1)

        loss_log = []
        reg_log = []
        for epoch in range(num_epochs):
            perm = np.random.permutation(n)
            loss = 0
            t = 0
            for i in perm:
                y = (x[i] @ w + xb * b).item()
                if c[i] * y < 1:
                    t += 1
                    lr = learning_rate * (1 + lam * t)
                    w = (1 - lam) * w + lr * c[i] * x[i]
                    b = (1 - lam) * b + lr * c[i] * xb
                loss += max(0, 1 - c[i] * y)
            loss /= n
            reg = lam/2 * (w * w).sum() + lam/2 * (b * b)
            loss_log.append(loss.item())
            reg_log.append(reg.item())
            print(f"epoch: {epoch}"
            f" loss: {loss_log[-1]:.3f}" 
            f" reg: {reg_log[-1]:.3f}"
            f" total: {loss_log[-1]+reg_log[-1]:.3f}")

            if epoch % 5 == 0 or epoch + 1 == num_epochs:
                plt.figure(1)
                plt.clf()
                plt.plot(loss_log)
                plt.plot(reg_log,'--')
                plt.plot([a+b for a, b in zip(loss_log, reg_log)])
                plt.legend(('loss', 'regularizer', 'total'))
                plt.pause(0.0001)

    return w, b

def svm_sdca(x, c, lam=0.01, epsilon=0.0005, num_epochs=1000):
    "Train an SVM using the SDCA algorithm."
    with torch.no_grad():
        xb = 1
        n = x.shape[0]
        d = x.shape[1]
        alpha = torch.zeros(n)
        w = torch.zeros(d)
        b = 0
        A = ((x * x).sum(1) + (xb * xb)) / (lam * n)

        lb_log = []
        ub_log = []
        lb = 0

        for epoch in range(num_epochs):
            perm = np.random.permutation(n)
            for i in perm:
                B = x[i] @ w + xb * b
                dalpha = (c[i] - B) / A[i]
                dalpha = c[i] * max(0, min(1, c[i] * (dalpha + alpha[i]))) - alpha[i]
                lb -= (A[i]/2 * (dalpha**2) + (B - c[i])*dalpha) / n
                w += (dalpha / (lam * n)) * x[i]
                b += (dalpha / (lam * n)) * xb
                alpha[i] += dalpha  

            scores = x @ w + xb * b
            ub = torch.clamp(1 - c * scores, min=0).mean() + (lam/2) * (w * w).sum() + (lam/2) * (b * b)
            lb_log.append(lb.item())
            ub_log.append(ub.item())        
            finish = (epoch + 1 == num_epochs) or (ub_log[-1] - lb_log[-1] < epsilon)

            if (epoch % 10 == 0) or finish:
                print(f"SDCA epoch: {epoch: 2d} lower bound: {lb_log[-1]:.3f} upper bound: {ub_log[-1]:.3f}")

            if ((epoch > 0) and (epoch % 200 == 0)) or finish:
                plt.figure(1)
                plt.clf()
                plt.title('SDCA optimization')
                plt.plot(lb_log)
                plt.plot(ub_log, '--')
                plt.legend(('lower bound', 'upper bound'))
                plt.xlabel('iteration')
                plt.ylabel('energy')
                plt.pause(0.0001)

            if finish:
                break

    return w, xb * b

def pr(labels, scores):
    "Plot the precision-recall curve."
    scores, perm = torch.sort(scores, descending=True)
    labels = labels[perm]
    tp = (labels > 0).to(torch.float32)
    ttp = torch.cumsum(tp, 0)
    precision = ttp / torch.arange(1, len(tp)+1, dtype=torch.float32)
    recall = ttp / tp.sum()
    ap = precision[tp > 0].mean()
    plt.plot(recall.numpy(), precision.numpy())
    plt.xlabel('recall')
    plt.ylabel('precision')
    return recall, precision, ap

def plot_ranked_list(imdb, pos, neg, perm):
    "Plot a ranked image list."
    n = 8
    all = torch.cat((pos,neg))
    for t in range(min(len(perm), n * n)):
        i = all[perm[t]]
        #print(i)
        image = get_image(imdb, i)
        plt.gcf().add_subplot(n,n,t+1)
        _, ax = imsc(image[0])
        ex = ax.get_extent()
        col = 'g' if i in pos else 'r'
        r = matplotlib.patches.Rectangle((ex[0],ex[3]), ex[1]-ex[0], ex[2]-ex[3], 
                                        linewidth=10, edgecolor=col, facecolor='none')
        plt.gca().add_patch(r)

def plot_saliency(model, layer, w, im):
    "Plot a saliency map."
    im.requires_grad_(True)
    y = model(im[None,:])[layer-1]
    y.backward(w.reshape(1,-1))
    saliency = torch.abs(im.grad.detach())
    saliency = torch.max(saliency,dim=0)[0][None,:]

    plt.clf()
    plt.gcf().add_subplot(1,2,1)
    imsc(im)
    plt.title('input image')

    plt.gcf().add_subplot(1,2,2)
    imsc(saliency)
    plt.title('class saliency')

class ImageCache():
    "Cache images from the Internet and compute their codes."
    def __init__(self):
        self.reset()
        self.model = get_encoder_cnn()

    def reset(self):
        "Empty the cache."
        self.cached = dict()

    def add(self, url):
        "Add the image at `url` to the cache."
        response = requests.get(url)
        image_pil = Image.open(io.BytesIO(response.content))
        image = self.model.normalize(image_pil)
        codes = self.model(image[None,:])
        self.cached[url] = {
            'image_pil': image_pil,
            'image': image,
            'codes': codes,
        }
    
    def get_codes(self, layer):
        "Get the codes for the cached images and the specified AlexNet `layer`."
        return torch.cat([x['codes'][layer-1] for x in self],0)

    def plot(self):
        "Plot the cache content."
        plt.clf()
        n = len(self)
        w = math.ceil(math.sqrt(n))
        h = (n + w - 1) // w
        for t, item in enumerate(self):
            plt.gcf().add_subplot(h,w,t+1)
            imsc(item['image'])

    def __len__(self):
        "Get the number of cached images."
        return len(self.cached)
    
    def __iter__(self):
        "Get an iterator over cached images and their codes."
        return iter(self.cached.values())



