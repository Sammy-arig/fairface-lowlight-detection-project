from ultralytics import YOLO
import pandas as pd
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load labels
labels = pd.read_csv("fairface_label_val.csv")

results = []

for index, row in labels.head(500).iterrows():   # use 500 images for faster results

    img_path = row["file"]      # already contains "val/1.jpg"
    race = row["race"]

    if not os.path.exists(img_path):
        continue

    # Run detection
    prediction = model(img_path)

    detected = len(prediction[0].boxes) > 0

    results.append({
        "race": race,
        "detected": detected
    })

# Convert to dataframe
df = pd.DataFrame(results)

# Calculate detection rate per race
group_results = df.groupby("race")["detected"].mean()

print("\nDetection rate by race:\n")
print(group_results)