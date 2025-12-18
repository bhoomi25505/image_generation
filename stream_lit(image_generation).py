import streamlit as st
import zipfile, os, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

st.title("Image Generation Playground")

prompt = st.text_input("Enter image prompt (concept name)")

zip_file = st.file_uploader("Upload Image ZIP", type="zip")

if zip_file:
    os.makedirs("data", exist_ok=True)
    with zipfile.ZipFile(zip_file) as z:
        z.extractall("data")

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ImageFolder("data", transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    st.success(f"Images loaded for prompt: {prompt}")

    # ================= AUTOENCODER =================
    autoencoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64, 256),
        nn.ReLU(),
        nn.Linear(256, 3*64*64),
        nn.Unflatten(1,(3,64,64))
    )

    ae_optim = optim.Adam(autoencoder.parameters(), lr=0.001)
    ae_loss = nn.MSELoss()

    if st.button("Train Autoencoder (1000 epochs)"):
        bar = st.progress(0)
        for epoch in range(1000):
            for x,_ in loader:
                recon = autoencoder(x)
                loss = ae_loss(recon, x)

                ae_optim.zero_grad()
                loss.backward()
                ae_optim.step()

            bar.progress((epoch+1)/1000)

        save_image(recon[:8], f"ae_{prompt}.png", normalize=True)
        st.image(f"ae_{prompt}.png")
        st.success("Autoencoder training complete")

    # ================= GAN =================
    G = nn.Sequential(
        nn.Linear(100,256),
        nn.ReLU(),
        nn.Linear(256,3*64*64),
        nn.Tanh()
    )

    D = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64,256),
        nn.ReLU(),
        nn.Linear(256,1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=0.0002)
    optD = optim.Adam(D.parameters(), lr=0.0002)

    if st.button("Train GAN (1000 epochs)"):
        bar = st.progress(0)
        for epoch in range(1000):
            for real,_ in loader:
                bs = real.size(0)

                # Train D
                z = torch.randn(bs,100)
                fake = G(z)

                d_loss = criterion(D(real), torch.ones(bs,1)) + \
                         criterion(D(fake.detach()), torch.zeros(bs,1))

                optD.zero_grad()
                d_loss.backward()
                optD.step()

                # Train G
                g_loss = criterion(D(fake), torch.ones(bs,1))
                optG.zero_grad()
                g_loss.backward()
                optG.step()

            bar.progress((epoch+1)/1000)

        z = torch.randn(8,100)
        samples = G(z).view(-1,3,64,64)
        save_image(samples, f"gan_{prompt}.png", normalize=True)
        st.image(f"gan_{prompt}.png")
        st.success("GAN training complete")

    # ================= DIFFUSION (TOY) =================
    def add_noise(x):
        return x + 0.1 * torch.randn_like(x)

    diffusion = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*64*64,256),
        nn.ReLU(),
        nn.Linear(256,3*64*64),
        nn.Unflatten(1,(3,64,64))
    )

    diff_optim = optim.Adam(diffusion.parameters(), lr=0.001)

    if st.button("Train Diffusion (1000 epochs)"):
        bar = st.progress(0)
        for epoch in range(1000):
            for x,_ in loader:
                noisy = add_noise(x)
                out = diffusion(noisy)
                loss = ((out - x)**2).mean()

                diff_optim.zero_grad()
                loss.backward()
                diff_optim.step()

            bar.progress((epoch+1)/1000)

        save_image(out[:8], f"diff_{prompt}.png", normalize=True)
        st.image(f"diff_{prompt}.png")
        st.success("Diffusion training complete")
