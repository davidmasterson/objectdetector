
import tensorflow as tf
import numpy as np
import cv2

# Path to the saved model
model_path = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
detect_fn = tf.saved_model.load(model_path)

with open('coco_labels.txt', 'r') as f:
    class_names = f.read().splitlines()

def load_image_into_numpy_array(path):
    # Load image with OpenCV
    image = cv2.imread(path)
    return image

def preprocess_image(image):
    # Convert image to tensor formatted image and add a batch dimension
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = tf.expand_dims(image, 0)
    return input_tensor

def detect_objects(image_path):
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = preprocess_image(image_np)
    detections = detect_fn(input_tensor)
    return image_np, detections

def display_detections(image_np, detections):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detection_classes = detections['detection_classes'].astype(np.int64)
    detection_boxes = detections['detection_boxes']
    detection_scores = detections['detection_scores']

    for i in range(num_detections):
        if detection_scores[i] > 0.5:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                          ymin * image_np.shape[0], ymax * image_np.shape[0])
            class_name = class_names[detection_classes[i]]
            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(image_np, f"{class_name}: {detection_scores[i]:.2f}", 
                        (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detection', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace with your image path
image_path = "./images/motorcycle.jpg"
confidence_threshold = 0.75
image_np, detections = detect_objects(image_path)
display_detections(image_np, detections)
