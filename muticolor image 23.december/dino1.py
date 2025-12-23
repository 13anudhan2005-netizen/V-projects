import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load image from URL
def load_image_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

# Load Dino image
dino_url = "https://cdn.mos.cms.futurecdn.net/DH4kS2UqnQ7Ckrs9z6yuAX-970-80.jpg.webp"
dino = load_image_from_url(dino_url).convert("RGB")
dino_np = np.array(dino)

# Split RGB channels
R, G, B = dino_np[:, :, 0], dino_np[:, :, 1], dino_np[:, :, 2]

# Create images emphasizing each channel
red_img = np.zeros_like(dino_np)
green_img = np.zeros_like(dino_np)
blue_img = np.zeros_like(dino_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display original and RGB color-emphasized images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(dino_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(red_img)
plt.title("Red Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(green_img)
plt.title("Green Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(blue_img)
plt.title("Blue Channel Emphasis")
plt.axis("off")

plt.tight_layout()
plt.show()

# Optional: Apply a colormap to grayscale
dino_gray = dino.convert("L")
dino_gray_np = np.array(dino_gray)

plt.figure(figsize=(6, 5))
plt.imshow(dino_gray_np, cmap="viridis")
plt.title("Colormapped Grayscale")
plt.axis("off")
plt.colorbar()
plt.show()
