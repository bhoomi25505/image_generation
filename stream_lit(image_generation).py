import streamlit as st
import zipfile, os, torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

st.set_page_config(page_title="AI Image Lab", layout="wide")
st.title("🎨 Image Generation Playground")

# --- 1. CACHED MODEL INITIALIZATION ---
# Using @st.cache_resource ensures models are only built once
@st.cache_resource
def load_models():
    # Autoencoder
    ae = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64, 256),
        nn.ReLU(),
        nn.Linear(256, 3*64*64),
        nn.Unflatten(1,(3,64,64))
    )
    
    # GAN Generator
    gan_g = nn.Sequential(
        nn.Linear(100,256),
        nn.ReLU(),
        nn.Linear(256,3*64*64),
        nn.Tanh()
    )
    
    # Toy Diffusion
    diff = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64,256),
        nn.ReLU(),
        nn.Linear(256,3*64*64),
        nn.Unflatten(1,(3,64,64))
    )
    return ae, gan_g, diff

autoencoder, G, diffusion = load_models()

# --- 2. DATA LOADING ---
zip_file = st.sidebar.file_uploader("Upload Image ZIP", type="zip")

if zip_file:
    # Extract only if folder doesn't exist to save time
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("data")

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    try:
        dataset = ImageFolder("data", transform=transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        st.sidebar.success(f"Loaded {len(dataset)} images.")
        
        # --- 3. UI TABS ---
        tab1, tab2, tab3 = st.tabs(["Autoencoder", "GAN", "Diffusion"])

        with tab1:
            if st.button("Run Reconstruction"):
                x, _ = next(iter(loader))
                with torch.no_grad():
                    recon = autoencoder(x)
                save_image(recon, "ae.png", normalize=True)
                st.image("ae.png", caption="Reconstructed Images")

        with tab2:
            if st.button("Generate from Noise"):
                z = torch.randn(8,100)
                with torch.no_grad():
                    fake = G(z).view(-1,3,64,64)
                save_image(fake, "gan.png", normalize=True)
                st.image("gan.png", caption="GAN Generated Images")

        with tab3:
            if st.button("Denoise Test"):
                x, _ = next(iter(loader))
                noisy = x + 0.1 * torch.randn_like(x)
                with torch.no_grad():
                    out = diffusion(noisy)
                save_image(out, "diff.png", normalize=True)
                st.image("diff.png", caption="Diffusion Output")
                
    except RuntimeError:
        st.error("Error: Ensure your ZIP contains a folder with images inside (e.g., 'birds/img1.jpg').")
