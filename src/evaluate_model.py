import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from data_augmentation import augment_data
from load_dataset import load_data


def evaluate_model(model, X_test, y_test_cat, categories):
    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)

    # Classification report
    report = classification_report(
        y_true_classes, y_pred_classes, target_names=categories
    )
    print(report)

    # Save report
    with open("metrics/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    data_dir = "data/Training"
    categories = ["notumor", "pituitary", "meningioma", "glioma"]
    data, labels = load_data(data_dir, categories)
    _, _, X_test, _, y_test_cat = augment_data(data, labels, categories)
    # Load the saved model
    model = load_model("models/best_model.keras")
    evaluate_model(model, X_test, y_test_cat, categories)
