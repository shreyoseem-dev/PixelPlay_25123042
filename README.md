# Pixel Play

**Author:** Shreyosee Mondal_25123042
**Challenge:** Pixel Play '26 - Video Anomaly Detection

## 📄 Project Overview
This repository contains the solution for the Pixel Play anomaly detection challenge. The objective was to identify abnormal events (fighting, running, accidents) in surveillance footage. Given the lack of granular frame-level labels for training, I implemented an **Unsupervised Hybrid Ensemble** approach.

## 🧠 The Approach: Hybrid Statistical & Deep Feature Engineering
Since the dataset was limited, I avoided using standard pre-trained classifiers to prevent bias. Instead, I implemented a hybrid strategy that combines deep architectural features with robust statistical image analysis.

### Key Components:
1.  **Deep Spatial Feature Extraction (`ImprovedCNN`):**
    - I designed a custom CNN architecture (`ImprovedCNN`) initialized with Kaiming Normalization.
    - Instead of relying on potentially biased pre-trained weights, this module utilizes the **intrinsic inductive bias** of Convolutional Neural Networks to extract hierarchical spatial structures from the video frames.
    - This acts as a high-dimensional feature projector, highlighting spatial irregularities in the frame structure.

2.  **Statistical Anomaly Scoring:**
    - To complement the deep features, I implemented a heuristic scoring engine (`compute_feature_score`) that calculates:
        - **Variance Analysis:** Detects sudden intensity shifts typical of explosions or fast movement.
        - **Edge Density:** Uses gradient magnitude to identify chaotic visual texture (e.g., crowds fighting).
        - **Color Deviation:** Tracks statistical outliers in channel distribution.

3.  **Ensemble Inference Logic:**
    - `Final Score = (0.6 * CNN_Score) + (0.4 * Statistical_Score)`
    - This weighted ensemble ensures robust detection by balancing abstract deep features with interpretable statistical metrics.
    - **Temporal Smoothing:** Applied a Gaussian rolling window to enforce temporal consistency and reduce false positives from camera noise.

## 🛠️ Usage
This solution is provided as a Jupyter Notebook. To reproduce the results, simply open `pixelplay-baseline.ipynb` in Kaggle or Google Colab (with GPU enabled) and run.+
