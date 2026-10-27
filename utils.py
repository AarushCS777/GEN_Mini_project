import torch
import numpy as np
from PIL import Image
from model import Generator

def load_generator_from_checkpoint(checkpoint_path, nz=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Generator(nz=nz).to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netG.eval()
    return netG

def generate_fashion_images(netG, num_images=16, nz=100, seed=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_tensors = netG(noise).cpu()
        fake_tensors = (fake_tensors + 1) / 2.0  # Normalize to [0, 1]
        fake_tensors = torch.clamp(fake_tensors, 0, 1)

    images = []
    for i in range(num_images):
        img_np = (fake_tensors[i].squeeze().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np, mode='L'))
    return images