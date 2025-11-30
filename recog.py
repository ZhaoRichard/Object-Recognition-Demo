# Object Recognition Demo
import cv2
import numpy as np
import os
import sys

# -------------------------------
# 1. Settings
# -------------------------------

input_path = "input.png"
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"

CONF_THRESH = 0.5
NMS_THRESH = 0.4


# -------------------------------
# Load class names from coco.names
# -------------------------------

names_path = "coco.names"

if not os.path.isfile(names_path):
    print(f"[ERROR] Class name file '{names_path}' not found.")
    print("Download it from YOLOv3 repository:")
    print("https://github.com/pjreddie/darknet/blob/master/data/coco.names")
    sys.exit(1)

with open(names_path, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

print(f"[INFO] Loaded {len(CLASS_NAMES)} class names.")


# -------------------------------
# 2. Basic file checks
# -------------------------------

if not os.path.isfile(input_path):
    print(f"[ERROR] Input file '{input_path}' not found.")
    sys.exit(1)

if not os.path.isfile(cfg_path):
    print(f"[ERROR] YOLO config file '{cfg_path}' not found.")
    sys.exit(1)

if not os.path.isfile(weights_path):
    print(f"[ERROR] YOLO weights file '{weights_path}' not found.")
    sys.exit(1)

print("[INFO] Files found. Loading image...")

# -------------------------------
# 3. Load & downscale image to <= 1080p
# -------------------------------

image = cv2.imread(input_path)
if image is None:
    print(f"[ERROR] '{input_path}' could not be opened as an image.")
    sys.exit(1)

h, w = image.shape[:2]
print(f"[INFO] Original image size: {w} x {h}")


max_w, max_h = 1024, 768
scale = min(max_w / w, max_h / h, 1.0)  # never scale up

if scale < 1.0:
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"[INFO] Resized image to: {new_w} x {new_h}")
else:
    print("[INFO] Image is already <= 1080p, not resized.")

height, width = image.shape[:2]

# -------------------------------
# 4. Load YOLO network
# -------------------------------

print("[INFO] Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# If you have CUDA and built OpenCV with it, you can enable this:
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()

# Handle different OpenCV formats of getUnconnectedOutLayers()
# - could be [[200], [227], [254]] or [200, 227, 254] or np.array
unconnected = np.array(unconnected).flatten()
output_layers = [layer_names[i - 1] for i in unconnected]

print(f"[INFO] Output layers: {output_layers}")

# -------------------------------
# 5. Build blob & forward pass
# -------------------------------

print("[INFO] Running forward pass...")
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1 / 255.0,
    size=(416, 416),
    swapRB=True,
    crop=False
)
net.setInput(blob)
layer_outputs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

# -------------------------------
# 6. Parse YOLO detections
# -------------------------------

for output in layer_outputs:
    for det in output:
        scores = det[5:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])

        if conf > CONF_THRESH:
            cx = int(det[0] * width)
            cy = int(det[1] * height)
            w_box = int(det[2] * width)
            h_box = int(det[3] * height)

            x = int(cx - w_box / 2)
            y = int(cy - h_box / 2)

            boxes.append([x, y, w_box, h_box])
            confidences.append(conf)
            class_ids.append(class_id)

print(f"[INFO] Raw detections above conf={CONF_THRESH}: {len(boxes)}")

# -------------------------------
# 7. Apply NMS
# -------------------------------

indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

if len(indices) == 0:
    print("[INFO] No detections kept after NMS.")
else:
    print(f"[INFO] Detections after NMS: {len(indices)}")

# -------------------------------
# 8. Draw detections
# -------------------------------

for i in indices:
    # i may be: 12, [12], [[12]], np.array([12]), etc.
    if hasattr(i, "__len__"):
        # Flatten until it's a scalar
        while hasattr(i, "__len__") and len(i) > 0:
            i = i[0]
    idx = int(i)

    x, y, w_box, h_box = boxes[idx]
    cls_id = class_ids[idx]
    conf = confidences[idx]

    label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"

    cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# -------------------------------
# 9. Save and show result
# -------------------------------

output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"[INFO] Output saved to '{output_path}'")

# Try to display (may not work in headless/remote environments)
try:
    cv2.namedWindow("YOLO detections", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO detections", image)
    print("[INFO] Close the image window or press any key in it to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"[WARN] Could not display window: {e}")
    print("[INFO] You can still open 'output.jpg' manually.")
