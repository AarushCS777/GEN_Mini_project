import streamlit as st
import matplotlib.pyplot as plt
import os
import torch

from train import train_dcgan
from utils import generate_fashion_images, load_generator_from_checkpoint

st.set_page_config(page_title="DCGAN Fashion Generator", layout="wide")

st.title("👕 Fashion Image Generator using DCGAN")
st.markdown("Train a Deep Convolutional GAN on Fashion-MNIST or generate new clothing designs in real-time.")

# Sidebar Controls
st.sidebar.header("⚙️ Model Controls")
mode = st.sidebar.radio("Select Mode", ["Train Model", "Generate Images from Checkpoint"])

nz = st.sidebar.slider("Latent Vector Dimension (nz)", min_value=32, max_value=256, value=100, step=16)
lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0002, 0.0005, 0.001], value=0.0002)

CHECKPOINT_FILE = "dcgan_fashion.pt"

if mode == "Train Model":
    st.subheader("🏋️ Model Training & Loss Metrics")
    epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=50, value=5)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
    
    start_train = st.button("🚀 Start Training")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📈 Live Loss Curves")
        plot_spot = st.empty()
        
    with col2:
        st.markdown("### 🖼️ Epoch Sample Output")
        image_spot = st.empty()
        
    def on_epoch_progress(epoch, total_epochs, g_loss, d_loss, sample_images, history):
        progress = epoch / total_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch [{epoch}/{total_epochs}] | Generator Loss: {g_loss:.4f} | Discriminator Loss: {d_loss:.4f}")
        
        # Plot Loss Curves
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["g_loss"], label="Generator Loss", color="blue")
        ax.plot(history["d_loss"], label="Discriminator Loss", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("DCGAN Adversarial Loss Progression")
        ax.legend()
        ax.grid(True)
        plot_spot.pyplot(fig)
        plt.close(fig)
        
        # Plot Sample Grid
        fig_img, axes = plt.subplots(4, 4, figsize=(4, 4))
        for i, ax_i in enumerate(axes.flatten()):
            ax_i.imshow(sample_images[i].squeeze(), cmap='gray')
            ax_i.axis('off')
        plt.tight_layout()
        image_spot.pyplot(fig_img)
        plt.close(fig_img)

    if start_train:
        st.info("Training initialized. Fetching dataset and training DCGAN...")
        netG, history = train_dcgan(
            epochs=epochs, 
            lr=lr, 
            nz=nz, 
            batch_size=batch_size, 
            progress_callback=on_epoch_progress,
            checkpoint_path=CHECKPOINT_FILE
        )
        st.success("Training complete! Model weights saved to checkpoint.")

elif mode == "Generate Images from Checkpoint":
    st.subheader("🖼️ Generate New Clothing Designs")
    
    num_to_gen = st.sidebar.slider("Number of Images to Generate", min_value=4, max_value=32, value=16, step=4)
    seed = st.sidebar.number_input("Random Seed (Optional)", value=42, step=1)
    
    if not os.path.exists(CHECKPOINT_FILE):
        st.warning("⚠️ No trained checkpoint file found. Please train the model first using the 'Train Model' tab.")
    else:
        try:
            netG = load_generator_from_checkpoint(CHECKPOINT_FILE, nz=nz)
            
            if st.button("🎲 Generate Batch"):
                generated_imgs = generate_fashion_images(netG, num_images=num_to_gen, nz=nz, seed=seed)
                
                cols = st.columns(4)
                for idx, img in enumerate(generated_imgs):
                    with cols[idx % 4]:
                        st.image(img, caption=f"Item #{idx+1}", use_column_width=True)
                        
        except Exception as e:
            st.error(f"Error loading model checkpoint: {e}")