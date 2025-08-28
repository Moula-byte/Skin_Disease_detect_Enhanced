from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import datetime
import csv
import matplotlib.pyplot as plt
import re

app = Flask(__name__)
model = load_model("skin_disease_model.h5")

class_map = {
    0: "Melanoma",
    1: "Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Benign Keratosis",
    5: "Dermatofibroma",
    6: "Vascular Lesion"
}

dataset_map = {
    "hmnist_8_8_L": "datasets/hmnist_8_8_L.csv",
    "hmnist_8_8_RGB": "datasets/hmnist_8_8_RGB.csv",
    "hmnist_28_28_L": "datasets/hmnist_28_28_L.csv",
    "hmnist_28_28_RGB": "datasets/hmnist_28_28_RGB.csv"
}

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam

def overlay_gradcam(original_img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(original_img * 255), 0.6, heatmap, 0.4, 0)
    return overlay

def predict_image(img_path, name, age, location):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array_exp)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = class_map.get(predicted_class, "Unknown")
    confidence = np.max(prediction)

    if confidence < 0.6:
        label += " — low confidence, consider dermatologist review"

    heatmap = make_gradcam_heatmap(img_array_exp, model)
    overlay = overlay_gradcam(img_array, heatmap)
    gradcam_path = os.path.join("static", "gradcam_" + os.path.basename(img_path))
    cv2.imwrite(gradcam_path, overlay)

    log_path = "prediction_log.csv"
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, os.path.basename(img_path), label, f"{confidence:.2f}", age, location, datetime.datetime.now()])

    return label, confidence, gradcam_path

def visualize_hmnist_sample(csv_path, save_path="static/sample.png"):
    df = pd.read_csv(csv_path)
    sample = df.sample(1).iloc[0]
    pixels = sample.drop("label").values
    size = int(np.sqrt(len(pixels) // 3)) if "RGB" in csv_path else int(np.sqrt(len(pixels)))
    if "RGB" in csv_path:
        img = np.array(pixels).reshape(size, size, 3).astype(np.uint8)
    else:
        img = np.array(pixels).reshape(size, size).astype(np.uint8)

    plt.imshow(img, cmap="gray" if "L" in csv_path else None)
    plt.axis("off")
    plt.title(f"Label: {sample['label']}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    sample_path = None
    error = None
    if request.method == "POST":
        file = request.files["image"]
        name = request.form.get("name")
        age_input = request.form.get("age")
        location = request.form.get("location")
        selected_dataset = request.form.get("dataset")

        # Validate name (only letters and spaces)
        if not re.match(r"^[A-Za-z\s]+$", name):
            error = "Please enter a valid patient name using letters and spaces only."
            return render_template("index.html", error=error)

        # Validate age
        try:
            age = int(age_input)
            if age <= 0 or age > 120:
                raise ValueError("Age must be between 1 and 120")
        except ValueError:
            error = "Please enter a valid numeric age between 1 and 120."
            return render_template("index.html", error=error)

        if selected_dataset:
            csv_path = dataset_map[selected_dataset]
            visualize_hmnist_sample(csv_path)
            sample_path = "static/sample.png"

        if file:
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
            filepath = os.path.join("static", filename)
            file.save(filepath)

            label, confidence, gradcam_path = predict_image(filepath, name, age, location)
            result = f"{label} ({confidence:.2f} confidence)"
            return render_template("index.html", prediction=result, image_path=filepath,
                                   gradcam_path=gradcam_path, name=name, age=age, location=location,
                                   sample_path=sample_path, selected_dataset=selected_dataset)
    return render_template("index.html", error=error)
if __name__ == "__main__":
    app.run(debug=True)