# Fashion Image Generator using DCGAN

A full-stack Deep Convolutional Generative Adversarial Network (DCGAN) application that generates Fashion-MNIST styled clothing images.

## Features
- **Frontend**: Streamlit UI with controls for hyperparameters (learning rate, latent vector dim, batch size, epoch counts), seed control, live training visualizer, and generated image gallery.
- **Backend**: PyTorch implementation of DCGAN with modular Generator, Discriminator, custom weight initialization, dataloader pipeline, and model checkpointing.
- **Visual Diagnostics**: Real-time G & D loss curves and adversarial progress monitoring.

## Local Setup Instructions

1. **Clone or Download the repository.**
2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt