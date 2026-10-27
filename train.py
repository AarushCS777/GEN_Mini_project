import os
import shutil
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Generator, Discriminator, weights_init

def ensure_fashion_mnist_downloaded(data_dir='./data'):
    """Downloads and verifies Fashion-MNIST binary files directly to bypass torchvision URL errors."""
    raw_folder = os.path.join(data_dir, "FashionMNIST", "raw")
    os.makedirs(raw_folder, exist_ok=True)

    # Directly pointing to raw GitHub mirror files
    files = {
        "train-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz"
    }

    for filename, url in files.items():
        file_path = os.path.join(raw_folder, filename)
        
        # Download file if missing or if file size is too small (e.g. invalid HTML error page)
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 10000:
            print(f"Downloading {filename} manually...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

def get_dataloader(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # 1. Download verified raw dataset files manually first
    ensure_fashion_mnist_downloaded(data_dir=data_dir)
    
    # 2. Set download=True! Since files are already in raw/, it skips the HTTP download 
    # but still executes the critical unzip & tensor conversion into the processed/ folder.
    dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def train_dcgan(epochs, lr, nz, batch_size, progress_callback=None, checkpoint_path="dcgan_fashion.pt"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = get_dataloader(batch_size=batch_size)
    
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    history = {"g_loss": [], "d_loss": []}
    
    for epoch in range(epochs):
        errD_epoch = 0.0
        errG_epoch = 0.0
        num_batches = len(dataloader)
        
        for i, data in enumerate(dataloader):
            # (1) Update D network
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
            
            output_real = netD(real_cpu)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()
            
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # (2) Update G network
            netG.zero_grad()
            output_g = netD(fake)
            errG = criterion(output_g, label_real)
            errG.backward()
            optimizerG.step()
            
            errD_epoch += errD.item()
            errG_epoch += errG.item()
        
        avg_d_loss = errD_epoch / num_batches
        avg_g_loss = errG_epoch / num_batches
        
        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)
        
        with torch.no_grad():
            sample_images = netG(fixed_noise).detach().cpu()
            sample_images = (sample_images + 1) / 2.0
            
        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_g_loss, avg_d_loss, sample_images, history)
            
    torch.save({
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'nz': nz,
        'history': history
    }, checkpoint_path)
    
    return netG, history