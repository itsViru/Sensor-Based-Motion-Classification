# 🚶‍♂️ Sensor-Based Motion Classification: Gait Analysis & Classification via Inertial Sensor Processing

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Signal Processing](https://img.shields.io/badge/Signal_Processing-Butterworth_Filters-green.svg)
![Validation](https://img.shields.io/badge/Validation-5--Fold_CV-red.svg)

## 📌 Project Overview
This repository contains the data processing pipeline, test strategy, and machine learning implementation for the binary classification of human gait (normal walking vs. walking with physical impairment)[cite: 3]. 

By capturing raw inertial measurement data from smartphone sensors (Accelerometer and Gyroscope), this project demonstrates end-to-end capabilities in **hardware data acquisition, signal processing, automated testing, and algorithmic evaluation**[cite: 1, 3].

> **🎯 Note for Recruiters:** 
> If you are reviewing this repository for roles in **Test Automation, Embedded Systems, or Hardware Testing**, please note the strong emphasis on rigorous validation methodologies (Cross-Validation), automated evaluation of sensor data, anomaly detection, and Python-based data pipelines.

## 🛠️ Tech Stack & Methodologies
*   **Languages & Libraries:** Python, TensorFlow / Keras, NumPy, Pandas.
*   **Sensor Interfacing:** Processing raw 3-axis accelerometer and gyroscope data captured via the Phyphox platform[cite: 1, 3].
*   **Signal Processing:** Frequency analysis, Low-pass Butterworth filtering, sequence extraction[cite: 1, 4].
*   **Validation Frameworks:** 5-Fold Subject-Wise Cross-Validation, Confusion Matrices, Loss/Accuracy tracking[cite: 1, 4].

---

## 📡 Sensor Data Acquisition & Signal Processing
A critical component of this project is handling noisy, real-world data from physical measurement devices. 

1.  **Hardware Interfacing:** Data was collected via smartphone inertial sensors placed at the lower back, capturing 3-axis acceleration and rotation rates[cite: 1, 3].
2.  **Filter Design:** To remove high-frequency noise and prevent phase distortion, a **Low Pass Butterworth Filter** was applied[cite: 1, 4]. This filter was chosen specifically for its "maximally flat" frequency response in the passband[cite: 1].
3.  **Anomaly Detection & Outlier Removal:** Implemented a Standard Deviation method, establishing a 3-standard-deviation cutoff ($\approx 99.7\%$ retention) to automatically identify and drop malicious sequences and extreme outliers from the Gaussian-like distribution[cite: 1].
4.  **Data Synchronization & Resampling:** Addressed sequence inconsistencies by segmenting the data into equal-length frames, utilizing a specific hop size to manage overlap and avoid data loss between frames[cite: 1, 4].

---

## 🧪 Automated Testing Strategy & Validation
To ensure the model is robust and suitable for real-world application, strict testing and validation protocols were implemented, mirroring best practices in software and hardware test automation.

*   **5-Fold Subject-Wise Cross-Validation:** To prevent data leakage and artificially inflated accuracy, the dataset was split subject-wise rather than sample-wise[cite: 1, 4]. This ensures the algorithm generalizes to *new* users rather than memorizing known subjects.
*   **Evaluation Metrics:** The system's performance was automatically evaluated tracking sparse categorical crossentropy loss and accuracy[cite: 1]. 
*   **False Positive / False Negative Analysis:** Confusion matrices were generated to deeply analyze the classification performance (e.g., yielding 180 True Positives, 7 False Positives, 1 False Negative, and 137 True Negatives in the final epoch evaluation)[cite: 1].
*   **Unlabeled Data Testing:** The final model was subjected to a blind test using unlabeled data from the same source to verify absolute prediction reliability[cite: 1].

---

## 🧠 Neural Network Architecture
The core classification was handled by building and tuning a Deep Neural Network via `tf.keras`. 

*   **Architecture:** The Standard Neural Network (SNN) features a flattened input layer, two hidden layers (100 neurons each), and a 2-neuron output layer[cite: 1].
*   **Activation Functions:** 
    *   Hidden Layers: ReLU ($R(z) = \max(0, z)$) for training efficiency[cite: 1].
    *   Output Layer: Sigmoid ($\sigma(z) = \frac{1}{1+e^{-z}}$) tailored for binary classification[cite: 1].
*   **Hyperparameter Optimization:** Iterative tuning was performed on the learning rate (optimal at 0.001 using the Adam optimizer), epoch count, and neuron density to prevent overfitting and stabilize the loss gradient[cite: 1].

---

## 📊 Results & Conclusion
Both sensor types proved highly effective for identifying gait impairments following the automated data processing pipeline.

*   **Accelerometer Model:** Achieved **~97.5% Accuracy** with a computed loss of ~0.084[cite: 1].
*   **Gyroscope Model:** Achieved **~98.1% Accuracy** with a computed loss of ~0.068[cite: 1].
*   **Model Comparison:** The refined Standard Neural Network (SNN) outperformed an experimental Convolutional Neural Network (CNN) in overall predictability and training stability[cite: 1].

*(Note: Data files and raw Python scripts are omitted from this repository due to file size constraints and data protection guidelines, but the logic and methodology are thoroughly documented above.)*
