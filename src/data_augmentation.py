from load_dataset import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def augment_data(data, labels, categories):
    # Normalize data
    data = data / 255.0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=len(categories))
    y_test_cat = to_categorical(y_test, num_classes=len(categories))

    # Initialize data generator
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    return datagen, X_train, X_test, y_train_cat, y_test_cat


if __name__ == "__main__":
    data_dir = "data/Training"
    categories = ["notumor", "pituitary", "meningioma", "glioma"]
    data, labels = load_data(data_dir, categories)
    augment_data(data, labels, categories)
