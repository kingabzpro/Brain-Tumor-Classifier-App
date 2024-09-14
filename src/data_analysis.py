import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_dataset import load_data


def preprocess_and_analyze(data, labels, categories):
    # Create a DataFrame for analysis
    df = pd.DataFrame({"label": labels})
    df["label_name"] = df["label"].apply(lambda x: categories[x])

    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x="label_name", data=df)
    plt.title("Class Distribution")
    plt.xlabel("Tumor Type")
    plt.ylabel("Count")
    plt.savefig("metrics/class_distribution.png")
    plt.close()

    # Display sample images from each category
    fig, axs = plt.subplots(1, len(categories), figsize=(20, 5))
    for i, category in enumerate(categories):
        sample_idx = np.where(labels == i)[0][0]
        axs[i].imshow(data[sample_idx].astype("uint8"))
        axs[i].set_title(category)
        axs[i].axis("off")
    plt.savefig("metrics/sample_images.png")
    plt.close()


if __name__ == "__main__":
    data_dir = "data/Training"
    categories = ["notumor", "pituitary", "meningioma", "glioma"]
    data, labels = load_data(data_dir, categories)
    preprocess_and_analyze(data, labels, categories)
