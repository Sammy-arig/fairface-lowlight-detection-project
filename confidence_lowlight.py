from ultralytics import YOLO
import pandas as pd
import os
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load labels
labels = pd.read_csv("fairface_label_val.csv")

results = []

for index, row in labels.head(500).iterrows():

    img_path = row["file"]
    race = row["race"]

    if not os.path.exists(img_path):
        continue

    # Read image
    img = cv2.imread(img_path)

    # Simulate low-light
    dark_img = (img * 0.2).astype("uint8")

    # Run detection on dark image
    prediction = model(dark_img, verbose=False)

    boxes = prediction[0].boxes

    if boxes is not None and len(boxes) > 0:

        confidence = float(boxes.conf.max())

    else:

        confidence = 0.0

    results.append({
        "race": race,
        "confidence": confidence
    })

df = pd.DataFrame(results)

confidence_results = df.groupby("race")["confidence"].mean().sort_values(ascending=False)

print("\nAverage LOW-LIGHT Detection Confidence by Race:\n")
print(confidence_results)