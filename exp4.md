#exp 4
# Object detection and recognition

from google.colab import files
uploaded = files.upload()

from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Perform object detection
results = model("input.jpg", save=True)

# Print detected objects and confidence
for r in results:
    for box in r.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf) * 100
        print(model.names[class_id], ":", round(confidence, 2), "%")

# Display output image
output_image = cv2.imread("runs/detect/predict/input.jpg")
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

plt.imshow(output_image)
plt.axis("off")

# Output :
<img width="1207" height="605" alt="image" src="https://github.com/user-attachments/assets/ff83aea8-0be5-4c47-b49e-f5dc49389cba" />
<img width="645" height="551" alt="image" src="https://github.com/user-attachments/assets/744f14db-0cf0-4c7d-9390-6d1d646e32e9" />
