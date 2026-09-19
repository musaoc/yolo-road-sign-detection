# Road Sign Object Detection — Fine-Tuning YOLOv8 & YOLO11

An end-to-end computer vision project that converts Pascal VOC annotations into YOLO format and fine-tunes state-of-the-art YOLOv8 and YOLO11 architectures for real-time road sign detection.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/finetuning-for-object-detection-simplified)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Computer%20Vision%20/%20Deep%20Learning-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[Road Sign Object Detection — Fine-Tuning YOLOv8 & YOLO11](https://www.kaggle.com/code/lazer999/finetuning-for-object-detection-simplified)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Data pipeline converting XML Pascal VOC annotations into normalized YOLO coordinate formats.
- Automated dataset splitting into stratified train/val/test splits with dataset YAML configuration.
- Fine-tuned Ultralytics YOLOv8n and YOLO11n on target road sign classes (TrafficLight, Stop, SpeedLimit, Crosswalk).
- Comparative metric benchmarking: precision, recall, mAP@0.5, and inference latency.
- Inference pipeline with bounding box visualization overlays on novel imagery.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart LR
    A[Pascal VOC XML & Imagery] --> B[Annotation Converter to YOLO format]
    B --> C[Dataset Split & YAML Configuration]
    C --> D[YOLOv8n Fine-Tuning]
    C --> E[YOLO11n Fine-Tuning]
    D --> F[mAP@50 & Latency Benchmark]
    E --> F
    F --> G[Real-Time Detection Engine]
```

---

## Repository Structure

```plaintext
yolo-road-sign-detection/
├── notebooks/
│   └── yolo-road-sign-detection.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/yolo-road-sign-detection.git
cd yolo-road-sign-detection
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/yolo-road-sign-detection.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [Road Sign Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle datasets download -d andrewmvd/road-sign-detection
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [Road Sign Object Detection — Fine-Tuning YOLOv8 & YOLO11](https://www.kaggle.com/code/lazer999/finetuning-for-object-detection-simplified)

If you found this project helpful or insightful, please consider starring the repository ⭐!
