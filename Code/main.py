import numpy as np
import os
import time
import copy
import pickle
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

from data_prep import prep_and_load_data
from model import get_model
import constants as CONST


def plotter(history_file):
    """Plot and save accuracy/loss graphs from training history."""
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(CONST.OUTPUT_DIR, 'accuracy.png'))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(CONST.OUTPUT_DIR, 'loss.png'))
    plt.show()
    plt.close()


def process_image(directory, img_path):
    """Read and preprocess a single image for prediction."""
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    if image is None:
        return None, None
    image_copy = copy.deepcopy(image)
    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image_std = image.astype('float32') / 255.0
    return image_copy, image_std


def video_write(model):
    """Generate a prediction video from test images."""
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("./prediction.mp4", fourcc, 1.0, (400, 400))
    val_map = {1: 'Dog', 0: 'Cat'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    image_paths = os.listdir(CONST.TEST_DIR)[:200]
    for count, img_path in enumerate(image_paths, 1):
        image, image_std = process_image(CONST.TEST_DIR, img_path)
        if image is None:
            continue

        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
        pred = model.predict(image_std, verbose=0)
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        s = f"{val_map[arg_max[0]]} - {max_val[0]*100:.2f}%"
        cv2.putText(image, s, location, font, fontScale, fontColor, lineType)

        frame = cv2.resize(image, (400, 400))
        out.write(frame)
        print(f"Prediction {count}/200")

    out.release()
    print("Video saved: prediction.mp4")


if __name__ == "__main__":
    # Load and split data
    data = prep_and_load_data()
    train_size = int(len(data) * CONST.SPLIT_RATIO)
    print(f"Total: {len(data)}, Train: {train_size}, Val: {len(data) - train_size}")

    train_data = data[:train_size]
    train_images = np.array([i[0] for i in train_data], dtype=np.float32)
    train_labels = np.array([i[1] for i in train_data], dtype=np.float32)

    test_data = data[train_size:]
    test_images = np.array([i[0] for i in test_data], dtype=np.float32)
    test_labels = np.array([i[1] for i in test_data], dtype=np.float32)

    del data, train_data, test_data
    print("Data loaded and split.")

    # Train
    model = get_model()
    tensorboard = TensorBoard(log_dir=os.path.join('logs', str(int(time.time()))))

    print("Training started...")
    history = model.fit(
        train_images, train_labels,
        batch_size=50, epochs=15, verbose=1,
        validation_data=(test_images, test_labels),
        callbacks=[tensorboard]
    )
    print("Training complete.")

    # Save model and history
    os.makedirs(CONST.MODEL_DIR, exist_ok=True)
    model.save(os.path.join(CONST.MODEL_DIR, 'cats_vs_dogs.h5'))

    os.makedirs(CONST.OUTPUT_DIR, exist_ok=True)
    history_file = os.path.join(CONST.OUTPUT_DIR, 'history.pickle')
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    # Plot and generate video
    plotter(history_file)
    video_write(model)
