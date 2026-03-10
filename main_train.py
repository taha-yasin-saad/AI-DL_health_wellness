import os
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientNetB1 COVID-19 Chest X-ray Classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory containing train/val/test folders",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=240,  # EfficientNetB1 default
        help="Input image size (img_size x img_size)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "focal", "focal_aug"],
        help="Training mode: baseline / focal / focal_aug",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.txt",
        help="File to save metrics summary",
    )
    return parser.parse_args()


def build_generators(data_dir, img_size, batch_size, mode):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Base augmentation for all modes
    if mode == "baseline" or mode == "focal":
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
        )
    else:
        # focal_aug: stronger augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            vertical_flip=True,
        )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_model(img_size, num_classes):
    base_model = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

    base_model.trainable = True  # fine-tune whole network; you can freeze some layers if GPU is weak

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)

    # Strategy II style head: BN + Dense + Dropout + L2 regularization
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def focal_loss(gamma=2.0, alpha=None):
    """
    Multi-class focal loss.
    alpha: 1D tensor of shape (num_classes,) with class weights, or None.
    """

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)

        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true * alpha, axis=-1)
        else:
            alpha_t = 1.0

        fl = alpha_t * tf.pow(1.0 - p_t, gamma) * ce
        return fl

    return loss_fn


def compute_class_alpha(train_gen, num_classes):
    labels = train_gen.classes  # numpy array of labels
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    inv_freq = 1.0 / (counts + 1e-6)
    alpha = inv_freq / np.sum(inv_freq)
    return tf.constant(alpha, dtype=tf.float32)


def train_and_evaluate(args):
    # Setup
    train_gen, val_gen, test_gen = build_generators(
        args.data_dir, args.img_size, args.batch_size, args.mode
    )
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    print("Class indices:", train_gen.class_indices)

    model = build_model(args.img_size, num_classes)

    initial_lr = 1e-4
    optimizer = optimizers.Adam(learning_rate=initial_lr)

    # Choose loss
    if args.mode == "baseline":
        loss_fn = "categorical_crossentropy"
    else:
        alpha = compute_class_alpha(train_gen, num_classes)
        print("Focal-loss alpha (class weights):", alpha.numpy())
        loss_fn = focal_loss(gamma=2.0, alpha=alpha)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Callbacks
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"best_effnetb1_{args.mode}.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[lr_scheduler, checkpoint],
    )

    # Evaluate on test set using best weights
    model.load_weights(f"best_effnetb1_{args.mode}.h5")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Detailed metrics
    y_true = test_gen.classes
    y_prob = model.predict(test_gen)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # Save full model
    model.save(f"covid_cxr_effnetb1_tf_{args.mode}")

    # Save metrics to file
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(f"\n=== MODE: {args.mode} ===\n")
        f.write(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n")

    print(f"\nResults appended to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if "/" in args.output else None
    train_and_evaluate(args)
