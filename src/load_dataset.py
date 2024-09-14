import os

import numpy as np
from PIL import Image


def load_data(data_dir, categories):
    # Load images and labels
    data = []
    labels = []
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                data.append(np.array(img))
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    data_dir = "data/Training"
    categories = ["notumor", "pituitary", "meningioma", "glioma"]
    data, labels = load_data(data_dir, categories)
    # Save data and labels if needed
