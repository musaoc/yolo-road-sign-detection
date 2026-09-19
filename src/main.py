"""
Road Sign Object Detection — Fine-Tuning YOLOv8 & YOLO11
An end-to-end computer vision project that converts Pascal VOC annotations into YOLO format and fine-tunes state-of-the-art YOLOv8 and YOLO11 architectures for real-time road sign detection.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/finetuning-for-object-detection-simplified
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 2 ---
import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import random
import xml.etree.ElementTree as ET # For parsing XML annotations
import matplotlib.pyplot as plt # For visualizing images

# --- Cell 3 ---
# Define the root directory where your raw 'images' and 'annotations' folders are located.
# IMPORTANT: Replace 'path/to/your_raw_dataset_root' with the actual path.
RAW_DATASET_ROOT = '/kaggle/input/road-sign-detection'
RAW_IMAGES_DIR = os.path.join(RAW_DATASET_ROOT, 'images')
RAW_ANNOTATIONS_DIR = os.path.join(RAW_DATASET_ROOT, 'annotations')

# Define the new root directory for the processed YOLO dataset.
# This is where the 'train', 'val', 'images', and 'labels' folders will be created.
PROCESSED_DATASET_ROOT = 'processed_yolo_dataset'
PROCESSED_DATA_YAML_PATH = os.path.join(PROCESSED_DATASET_ROOT, 'data.yaml')

# Define your custom classes and their mapping to integer IDs.
# Ensure these match the classes in your XML annotations.
CUSTOM_CLASSES = {
    'trafficlight': 0,
    'stop': 1,
    'speedlimit': 2,
    'crosswalk': 3
}
CLASS_NAMES = list(CUSTOM_CLASSES.keys()) # For data.yaml and visualization

# --- Cell 4 ---

# --- Configuration ---

# Training parameters
BASE_YOLO_MODEL = 'yolov8n.pt' # 'yolov8n.pt' (nano) is a good starting point
EPOCHS = 10
IMGSZ = 640
BATCH_SIZE = 32
PROJECT_NAME = 'sign_detection_project'
RUN_NAME = 'yolov8_fine_tuned_signs'

# Dataset split ratio
TRAIN_SPLIT_RATIO = 0.6 # 80%

# --- Cell 5 ---
# Path for a sample image to test inference after training
# IMPORTANT: Replace 'path/to/your_sample_image.jpg' with an actual image path
# For demonstration, we'll use a random image from the validation set after splitting.
SAMPLE_INFERENCE_IMAGE_PATH = None # Will be set after dataset split

# --- Helper Functions ---

def check_cuda_availability():
    """
    Checks if CUDA (NVIDIA GPU) is available and prints the status.
    Ultralytics YOLO automatically utilizes CUDA if detected.
    """
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("❌ CUDA is not available. Training will run on CPU, which will be slower.")
    print("-" * 50)

# --- Cell 6 ---



def parse_xml_annotation(xml_file_path: str, img_width: int, img_height: int) -> list:
    """
    Parses a PASCAL VOC XML annotation file and converts bounding box coordinates
    to YOLO format (normalized x_center, y_center, width, height).

    Args:
        xml_file_path (str): Path to the XML annotation file.
        img_width (int): Width of the corresponding image.
        img_height (int): Height of the corresponding image.

    Returns:
        list: A list of strings, where each string is a YOLO annotation line
              (e.g., "class_id x_center y_center width height").
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    yolo_annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CUSTOM_CLASSES:
            print(f"Warning: Class '{class_name}' not defined in CUSTOM_CLASSES. Skipping.")
            continue

        class_id = CUSTOM_CLASSES[class_name]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert to YOLO format (normalized)
        x_center = (xmin + xmax) / (2.0 * img_width)
        y_center = (ymin + ymax) / (2.0 * img_height)
        width = (xmax - xmin) / float(img_width)
        height = (ymax - ymin) / float(img_height)

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_annotations

def create_yolo_dataset_structure(
    raw_images_dir: str,
    raw_annotations_dir: str,
    processed_dataset_root: str,
    train_split_ratio: float
) -> str:
    """
    Creates the YOLO-compatible dataset structure (images/train, images/val,
    labels/train, labels/val) and generates the data.yaml file.

    Args:
        raw_images_dir (str): Path to the directory containing raw images.
        raw_annotations_dir (str): Path to the directory containing raw XML annotations.
        processed_dataset_root (str): Root directory for the new YOLO dataset structure.
        train_split_ratio (float): Ratio for splitting data into training and validation sets.

    Returns:
        str: Path to the generated data.yaml file.
    """
    print(f"--- Preparing Dataset ---")
    print(f"Raw Images: {raw_images_dir}")
    print(f"Raw Annotations: {raw_annotations_dir}")
    print(f"Processed Dataset Output: {processed_dataset_root}")
    print(f"Train/Val Split Ratio: {train_split_ratio:.2f}")
    print("-" * 50)

    # Clean up previous processed dataset if it exists
    if os.path.exists(processed_dataset_root):
        print(f"Removing existing processed dataset directory: {processed_dataset_root}")
        shutil.rmtree(processed_dataset_root)

    # Create new directories
    os.makedirs(os.path.join(processed_dataset_root, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_dataset_root, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(processed_dataset_root, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_dataset_root, 'labels', 'val'), exist_ok=True)

    image_files = [f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files) # Shuffle for random split

    train_files = image_files[:int(len(image_files) * train_split_ratio)]
    val_files = image_files[int(len(image_files) * train_split_ratio):]

    print(f"Total images found: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # Process and copy files
    for i, file_list in enumerate([train_files, val_files]):
        subset_name = 'train' if i == 0 else 'val'
        print(f"Processing {subset_name} set...")
        for img_filename in file_list:
            base_filename = os.path.splitext(img_filename)[0]
            xml_filename = base_filename + '.xml'
            
            raw_img_path = os.path.join(raw_images_dir, img_filename)
            raw_xml_path = os.path.join(raw_annotations_dir, xml_filename)

            if not os.path.exists(raw_xml_path):
                print(f"Warning: Annotation file not found for {img_filename}. Skipping.")
                continue

            # Read image to get its dimensions
            img = cv2.imread(raw_img_path)
            if img is None:
                print(f"Warning: Could not read image {img_filename}. Skipping.")
                continue
            img_height, img_width, _ = img.shape

            # Parse XML and get YOLO annotations
            yolo_annotations = parse_xml_annotation(raw_xml_path, img_width, img_height)

            if not yolo_annotations:
                print(f"Warning: No valid annotations found for {img_filename}. Skipping.")
                continue

            # Copy image
            shutil.copy(raw_img_path, os.path.join(processed_dataset_root, 'images', subset_name, img_filename))

            # Write YOLO annotation file
            yolo_label_path = os.path.join(processed_dataset_root, 'labels', subset_name, base_filename + '.txt')
            with open(yolo_label_path, 'w') as f:
                for line in yolo_annotations:
                    f.write(line + '\n')

    # Generate data.yaml with paths relative to the data.yaml file itself
    data_yaml_content = f"""
# YOLOv8 Dataset Configuration
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(PROCESSED_DATA_YAML_PATH, 'w') as f:
        f.write(data_yaml_content)

    print(f"✅ Dataset preparation complete. data.yaml generated at: {PROCESSED_DATA_YAML_PATH}")
    print("-" * 50)
    return PROCESSED_DATA_YAML_PATH
def visualize_sample_data(
    dataset_root: str,
    class_names: list,
    num_samples: int = 5,
    subset: str = 'train'
):
    """
    Visualizes a few sample images with their bounding box annotations and labels.

    Args:
        dataset_root (str): Root directory of the processed YOLO dataset.
        class_names (list): List of class names in order of their IDs.
        num_samples (int): Number of random samples to display.
        subset (str): Which subset to visualize ('train' or 'val').
    """
    print(f"--- Visualizing Sample Data from {subset} set ---")
    images_dir = os.path.join(dataset_root, 'images', subset)
    labels_dir = os.path.join(dataset_root, 'labels', subset)

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"❌ Error: Image or label directory not found for {subset} set.")
        return

    all_image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_image_files:
        print(f"No images found in {images_dir} to visualize.")
        return

    # Select random samples
    sample_files = random.sample(all_image_files, min(num_samples, len(all_image_files)))

    plt.figure(figsize=(15, 5 * num_samples))
    for i, img_filename in enumerate(sample_files):
        base_filename = os.path.splitext(img_filename)[0]
        img_path = os.path.join(images_dir, img_filename)
        label_path = os.path.join(labels_dir, base_filename + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib

        h, w, _ = img.shape

        # Read annotations
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                    annotations.append((class_id, x_center, y_center, bbox_width, bbox_height))
        else:
            print(f"No annotation file found for {img_filename}")

        # Draw bounding boxes
        for ann in annotations:
            class_id, x_center, y_center, bbox_width, bbox_height = ann
            
            # Convert normalized YOLO to pixel coordinates (x1, y1, x2, y2)
            x1 = int((x_center - bbox_width / 2) * w)
            y1 = int((y_center - bbox_height / 2) * h)
            x2 = int((x_center + bbox_width / 2) * w)
            y2 = int((y_center + bbox_height / 2) * h)

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

            # Put label text
            label = class_names[class_id]
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img)
        plt.title(f"Image: {img_filename} (Annotations: {len(annotations)})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("-" * 50)



# --- Cell 7 ---
def fine_tune_yolo_model(
    model_name: str,
    data_yaml_path: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    project_name: str,
    run_name: str
) -> str:
    """
    Loads a pre-trained YOLOv8 model and fine-tunes it on a custom dataset.

    Args:
        model_name (str): Name of the pre-trained YOLOv8 model (e.g., 'yolov8n.pt').
        data_yaml_path (str): Path to the dataset configuration file (data.yaml).
        epochs (int): Number of epochs for training.
        imgsz (int): Image size for training.
        batch_size (int): Batch size for training.
        project_name (str): Name of the project directory for saving results.
        run_name (str): Name of the specific training run.

    Returns:
        str: Path to the best trained model weights.
    """
    print(f"Starting YOLOv8 fine-tuning with {model_name}...")
    print(f"Dataset YAML: {data_yaml_path}")
    print(f"Epochs: {epochs}, Image Size: {imgsz}, Batch Size: {batch_size}")
    print(f"Project: {project_name}, Run: {run_name}")
    print("-" * 50)

    # Load a pre-trained YOLOv8 model
    model = YOLO(model_name)

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name=run_name,
        # You can add more arguments here, e.g.,
        # cache='ram' or 'disk' for faster data loading,
        # patience=50 for early stopping,
        # lr0=0.001 for initial learning rate (often lower for fine-tuning)
    )

    trained_model_path = os.path.join('runs', 'detect', run_name, 'weights', 'best.pt')
    print(f"\n✅ Fine-tuning complete! Best model saved at: {trained_model_path}")
    print("-" * 50)
    return trained_model_path

def predict_and_visualize(model_path: str, image_path: str, conf_threshold: float = 0.25):
    """
    Performs inference on a single image using the fine-tuned YOLOv8 model
    and visualizes the results.

    Args:
        model_path (str): Path to the trained YOLOv8 model weights.
        image_path (str): Path to the image file for inference.
        conf_threshold (float): Confidence threshold for displaying detections.
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Sample image for inference not found at {image_path}")
        return

    print(f"Loading fine-tuned model from: {model_path}")
    model = YOLO(model_path)

    print(f"Performing inference on: {image_path}")
    results = model.predict(source=image_path, conf=conf_threshold, save=True, show=False)

    for r in results:
        im_bgr = r.plot()
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        print(f"Inference results saved to: {r.save_dir}")
        print(f"Detected {len(r.boxes)} objects.")
        for box in r.boxes:
            c = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            print(f"  Class: {r.names[c]}, Confidence: {conf:.2f}, Bounding Box: {xyxy}")

        # Display the image with predictions
        plt.figure(figsize=(10, 8))
        plt.imshow(im_rgb)
        plt.title(f"Inference Result for {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()
    print("-" * 50)


# --- Cell 8 ---
check_cuda_availability()

# --- Cell 9 ---
try:
    # This function will create the 'processed_yolo_dataset' directory
    # and populate it with 'images/train', 'images/val', 'labels/train', 'labels/val'
    # and also generate the 'data.yaml' file.
    processed_data_yaml_path = create_yolo_dataset_structure(
        raw_images_dir=RAW_IMAGES_DIR,
        raw_annotations_dir=RAW_ANNOTATIONS_DIR,
        processed_dataset_root=PROCESSED_DATASET_ROOT,
        train_split_ratio=TRAIN_SPLIT_RATIO
    )
except FileNotFoundError as e:
    print(f"\nERROR: {e}. Please ensure RAW_DATASET_ROOT, RAW_IMAGES_DIR, and RAW_ANNOTATIONS_DIR are correct.")
    print("Make sure your raw 'images' and 'annotations' folders exist at the specified path.")
    exit()
except Exception as e:
    print(f"\nAn error occurred during dataset preparation: {e}")
    exit()

# Set a sample image for inference from the validation set
val_images_dir = os.path.join(PROCESSED_DATASET_ROOT, 'images', 'val')
if os.path.exists(val_images_dir) and len(os.listdir(val_images_dir)) > 0:
    SAMPLE_INFERENCE_IMAGE_PATH = os.path.join(val_images_dir, random.choice(os.listdir(val_images_dir)))
else:
    print("Warning: No images found in the validation set for inference testing.")
    SAMPLE_INFERENCE_IMAGE_PATH = None

# --- Cell 10 ---
visualize_sample_data(
    dataset_root=PROCESSED_DATASET_ROOT,
    class_names=CLASS_NAMES,
    num_samples=5, # Display 3 sample images
    subset='train'
)

# --- Cell 11 ---
try:
    trained_model_path = fine_tune_yolo_model(
        model_name=BASE_YOLO_MODEL,
        data_yaml_path='/kaggle/working/processed_yolo_dataset/data.yaml', # Use the dynamically generated path
        epochs=10,
        imgsz=300,
        batch_size=32,
        project_name=PROJECT_NAME,
        run_name=RUN_NAME
    )
except Exception as e:
    print(f"\nAn error occurred during fine-tuning: {e}")
    print("Please check your dataset path, data.yaml, and ensure Ultralytics is installed.")
    

# --- Cell 13 ---
try:
    trained_model_path = fine_tune_yolo_model(
        model_name='yolo11n.pt',
        data_yaml_path='/kaggle/working/processed_yolo_dataset/data.yaml', # Use the dynamically generated path
        epochs=15,
        imgsz=300,
        batch_size=64,
        project_name='SecondModelYolo11n',
        run_name=RUN_NAME
    )
except Exception as e:
    print(f"\nAn error occurred during fine-tuning: {e}")
    print("Please check your dataset path, data.yaml, and ensure Ultralytics is installed.")
    

# --- Cell 14 ---

# 5. Perform inference with the fine-tuned model on a sample image
SAMPLE_INFERENCE_IMAGE_PATH = os.path.join(val_images_dir, random.choice(os.listdir(val_images_dir)))
if SAMPLE_INFERENCE_IMAGE_PATH:
    predict_and_visualize('/kaggle/working/SecondModelYolo11n/yolov8_fine_tuned_signs/weights/best.pt', SAMPLE_INFERENCE_IMAGE_PATH)
else:
    print("\nSkipping inference as no validation images were found.")

# --- Cell 15 ---
if SAMPLE_INFERENCE_IMAGE_PATH:
    predict_and_visualize('/kaggle/working/sign_detection_project/yolov8_fine_tuned_signs/weights/best.pt', SAMPLE_INFERENCE_IMAGE_PATH)
else:
    print("\nSkipping inference as no validation images were found.")

# --- Cell 16 ---
def plot_comparison_metrics(model_results_paths: dict[str, str]):
    """
    Plots various object detection metrics for comparison across multiple YOLOv8 models.

    Args:
        model_results_paths (dict[str, str]): A dictionary where keys are model names
                                             (e.g., 'YOLOv8n', 'YOLOv8s_Run2') and values are
                                             paths to their respective 'results.csv' files.
    """
    print("--- Plotting Comparison Metrics ---")
    
    metrics_to_plot = {
        'metrics/mAP50(B)': 'mAP@0.5',
        'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall'
    }
    
    num_plots = len(metrics_to_plot) + 1 # +1 for F1-score
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    if num_plots == 1: # Handle case with single subplot
        axes = [axes]

    all_data = {}
    for model_name, results_csv_path in model_results_paths.items():
        if not os.path.exists(results_csv_path):
            print(f"❌ Warning: results.csv not found for {model_name} at {results_csv_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(results_csv_path)
            all_data[model_name] = df
            print(f"Loaded results for {model_name} from {results_csv_path}")
        except Exception as e:
            print(f"❌ Error loading {results_csv_path} for {model_name}: {e}. Skipping.")

    if not all_data:
        print("No valid results.csv files loaded for comparison.")
        return

    # Plot each metric
    for i, (col_name, title) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        for model_name, df in all_data.items():
            if col_name in df.columns:
                ax.plot(df['epoch'], df[col_name], label=model_name)
            else:
                print(f"Warning: Column '{col_name}' not found in {model_name}'s results.csv.")
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)

    # Plot F1-score
    ax_f1 = axes[num_plots - 1]
    for model_name, df in all_data.items():
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            precision = df['metrics/precision(B)']
            recall = df['metrics/recall(B)']
            # Avoid division by zero
            f1_score = 2 * (precision * recall) / (precision + recall).replace(0, 1e-9) 
            ax_f1.plot(df['epoch'], f1_score, label=model_name)
        else:
            print(f"Warning: Precision or Recall columns not found for F1-score calculation in {model_name}'s results.csv.")
    ax_f1.set_title('F1-score')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1-score')
    ax_f1.legend()
    ax_f1.grid(True)

    plt.tight_layout()
    plt.show()
    print("-" * 50)


# --- Cell 17 ---
import pandas as pd

# --- Cell 18 ---

model_comparison_paths = {
    'YOLOv11n_FineTuned': os.path.join('/kaggle/working/SecondModelYolo11n/yolov8_fine_tuned_signs', 'results.csv'),
    'YOLOv8n_FineTuned': os.path.join('/kaggle/working/sign_detection_project/yolov8_fine_tuned_signs', 'results.csv'),

}
plot_comparison_metrics(model_comparison_paths)





if __name__ == "__main__":
    print("Pipeline execution complete.")
