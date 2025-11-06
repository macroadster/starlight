# lsb_steganography.py - LSB steganography demonstration
import os
import cv2
import numpy as np


# Simple LSB steganography functions
def embed_lsb_simple(image_path, msg_bits):
    """Embed message bits into image using LSB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    # Flatten image and embed bits sequentially across all channels
    flat_img = img_rgb.reshape(-1)
    msg_flat = np.array(msg_bits, dtype=np.uint8)

    # Embed in LSB of all pixels sequentially
    for i in range(min(len(msg_flat), len(flat_img))):
        flat_img[i] = (flat_img[i] & 0xFE) | msg_flat[i]

    result = flat_img.reshape(h, w, c)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def extract_lsb_simple(image_path, msg_len=100):
    """Extract message bits from image using LSB (all channels sequential)"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = img_rgb.reshape(-1)

    bits = []
    for i in range(min(msg_len, len(flat_img))):
        bits.append(flat_img[i] & 1)

    return np.array(bits, dtype=int)


def embed(cover_path, msg_bits):
    """Embed message using LSB"""
    return embed_lsb_simple(cover_path, msg_bits)


def extract(stego_path):
    """Extract message using LSB"""
    return extract_lsb_simple(stego_path, 100)


def analyze_lsb(image_path):
    """Analyze LSB patterns for steganography detection"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Extract LSBs from each color channel
    lsb_r = (img[:, :, 0].astype(np.uint8) & 1).astype(np.float32)
    lsb_g = (img[:, :, 1].astype(np.uint8) & 1).astype(np.float32)
    lsb_b = (img[:, :, 2].astype(np.uint8) & 1).astype(np.float32)

    # Calculate LSB statistics
    total_pixels = img.shape[0] * img.shape[1]
    ones_r = np.sum(lsb_r)
    ones_g = np.sum(lsb_g)
    ones_b = np.sum(lsb_b)

    # Calculate entropy properly
    def binary_entropy(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    stats = {
        "total_pixels": total_pixels,
        "lsb_ones_ratio": {
            "red": float(ones_r / total_pixels),
            "green": float(ones_g / total_pixels),
            "blue": float(ones_b / total_pixels),
        },
        "lsb_entropy": {
            "red": float(binary_entropy(lsb_r.mean())),
            "green": float(binary_entropy(lsb_g.mean())),
            "blue": float(binary_entropy(lsb_b.mean())),
        },
    }

    return stats, (lsb_r, lsb_g, lsb_b)


def extract_lsb_message(image_path, msg_length=100):
    """Extract potential hidden message from LSBs"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Simple sequential LSB extraction
    lsb_bits = []
    for channel in range(3):  # RGB channels
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if len(lsb_bits) < msg_length:
                    lsb_bits.append(int((img[i, j, channel] >> 0) & 1))
                else:
                    break
            if len(lsb_bits) >= msg_length:
                break
        if len(lsb_bits) >= msg_length:
            break

    return np.array(lsb_bits[:msg_length])


# demo
if __name__ == "__main__":
    msg = np.random.randint(0, 2, 100)
    cover_path = "../val/clean/clean-0405.png"

    # Check if cover image exists
    if not os.path.exists(cover_path):
        print(f"Error: Cover image {cover_path} not found!")
        exit(1)

    print("=== LSB Steganography Demo ===")
    print("Embedding message...")
    stego_img = embed(cover_path, msg)
    cv2.imwrite("stego_demo.png", stego_img)
    print("Stego image saved as stego_demo.png")

    print("Extracting message...")
    rec = extract("stego_demo.png")
    print("Recovered bits match?", np.array_equal(msg, rec))
    print("Bit error rate:", np.mean(msg != rec))

    print("\n=== LSB Analysis Demo ===")
    print("Analyzing original cover image...")
    cover_stats, _ = analyze_lsb(cover_path)
    print(
        f"Cover LSB ratios: R={cover_stats['lsb_ones_ratio']['red']:.3f}, "
        f"G={cover_stats['lsb_ones_ratio']['green']:.3f}, "
        f"B={cover_stats['lsb_ones_ratio']['blue']:.3f}"
    )

    print("Analyzing stego image...")
    stego_stats, _ = analyze_lsb("stego_demo.png")
    print(
        f"Stego LSB ratios: R={stego_stats['lsb_ones_ratio']['red']:.3f}, "
        f"G={stego_stats['lsb_ones_ratio']['green']:.3f}, "
        f"B={stego_stats['lsb_ones_ratio']['blue']:.3f}"
    )

    print("\n=== Summary ===")
    print("LSB steganography successfully hides and recovers messages")
    print("The message is embedded in the least significant bits of image pixels")
    print("This method provides perfect recovery but limited robustness")
