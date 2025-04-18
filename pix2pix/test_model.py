import numpy as np
from matplotlib import pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0")

model_path = "/home2/zwang/pix2pix/checkpoint/xcat128/netG_model_epoch.pth"
net_g = torch.load(model_path).to(device)

x = np.load("/home2/zwang/pix2pix/atten_40KeV.npy")
gt = np.load("/home2/zwang/pix2pix/atten_140KeV.npy")

for i in range(100):
    img = x[i,:,:]
    img = img / 0.2319 * 2 - 1
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).float()
    img = img.unsqueeze(0)
    input = img.to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()
    image_numpy = out_img.float().numpy()
    #image_numpy = (image_numpy + 1) * 0.1216 / 2
    #image_numpy = (image_numpy + 1) * 0.0917 / 2
    #image_numpy = (image_numpy + 1) * 0.0790 / 2
    #image_numpy = (image_numpy + 1) * 0.0719 / 2
    image_numpy = (image_numpy + 1) * 0.0671 / 2

    target_img = gt[i,:,:]

    print(ssim(target_img,image_numpy.reshape([128,128])))

    fig = plt.figure(figsize=(80, 30))
    rows = 1
    cols = 3

    fig.add_subplot(rows, cols, 1)
    plt.imshow(target_img.reshape([128,128]), cmap='gray')
    plt.clim(0, 0.2)
    plt.axis('off')
    plt.title('target')

    fig.add_subplot(rows, cols, 2)
    plt.imshow(image_numpy.reshape([128,128]), cmap='gray')
    plt.clim(0, 0.2)
    plt.axis('off')
    plt.title('obtain')

    fig.add_subplot(rows, cols, 3)
    plt.imshow(target_img-image_numpy.reshape([128,128]), cmap='gray')
    plt.clim(0, 0.05)
    plt.axis('off')
    plt.title('diff')


    nom = '/home2/zwang/pix2pix/results/xcat128/40140/'+str(i)+'.png'
    plt.savefig(nom)



