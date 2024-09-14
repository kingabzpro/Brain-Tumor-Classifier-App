import datetime

from data_augmentation import augment_data
from load_dataset import load_data
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def build_and_train_model(
    datagen, X_train, y_train_cat, X_test, y_test_cat, categories
):
    # Load pre-trained EfficientNetB0 model
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Unfreeze some of the base model's layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(categories), activation="softmax")(x)

    # Create new model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Define callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_cb = ModelCheckpoint(
        filepath="models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )

    # Prepare data generator
    train_generator = datagen.flow(X_train, y_train_cat, batch_size=32)

    # Train model
    history = model.fit(
        train_generator,
        validation_data=(X_test, y_test_cat),
        epochs=20,
        callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb, lr_scheduler],
    )

    # Save model in TensorFlow SavedModel format (safely)
    model.save("models/saved_model.keras")

    return model, history


if __name__ == "__main__":
    data_dir = "data/Training"
    categories = ["notumor", "pituitary", "meningioma", "glioma"]
    data, labels = load_data(data_dir, categories)
    datagen, X_train, X_test, y_train_cat, y_test_cat = augment_data(
        data, labels, categories
    )
    build_and_train_model(datagen, X_train, y_train_cat, X_test, y_test_cat, categories)
