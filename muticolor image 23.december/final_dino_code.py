import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Dino Image Processor", layout="wide")

# Title
st.title("Dino Image - Multi-Color Channel Visualizer")

# Load image from URL
@st.cache_data
def load_image():
    url = "https://cdn.mos.cms.futurecdn.net/DH4kS2UqnQ7Ckrs9z6yuAX-970-80.jpg.webp"
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

# Load and display image
dino = load_image()
st.image(dino, caption="Original Dino Image", use_container_width=True)

# Convert to NumPy array
dino_np = np.array(dino)
R, G, B = dino_np[:, :, 0], dino_np[:, :, 1], dino_np[:, :, 2]

# Create channel images
red_img = np.zeros_like(dino_np)
green_img = np.zeros_like(dino_np)
blue_img = np.zeros_like(dino_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display RGB channels
st.subheader("RGB Channel Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(red_img, caption="Red Channel", use_container_width=True)

with col2:
    st.image(green_img, caption="Green Channel", use_container_width=True)

with col3:
    st.image(blue_img, caption="Blue Channel", use_container_width=True)

# Grayscale + Colormap
st.subheader("Colormapped Grayscale Image")

colormap = st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
)

dino_gray = dino.convert("L")
dino_gray_np = np.array(dino_gray)

# Plot using matplotlib with colormap
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(dino_gray_np, cmap=colormap)
ax.axis("off")

# Render in Streamlit
st.pyplot(fig)
