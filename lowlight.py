from ultralytics import YOLO
import pandas as pd
import cv2
import os

model = YOLO("yolov8n.pt")

labels = pd.read_csv("fairface_label_val.csv")

results = []

for index, row in labels.head(500).iterrows():

    img_path = row["file"]
    race = row["race"]

    if not os.path.exists(img_path):
        continue

    # read image
    img = cv2.imread(img_path)

    # simulate low light (reduce brightness)
    dark_img = (img * 0.2).astype("uint8")

    # run detection
    prediction = model(dark_img)

    detected = len(prediction[0].boxes) > 0

    results.append({
        "race": race,
        "detected": detected
    })

df = pd.DataFrame(results)

group_results = df.groupby("race")["detected"].mean()

print("\nLow-light detection rate by race:\n")
print(group_results)