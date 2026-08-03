# Gravitational Wave with Deep Learning Project

Welcome to the **Gravitational Wave with Deep Learning Project** repository!  
This repository was my final project for my study in pyhsics. This repository contains a complete Deep Learning pipeline for analyzing **Gravitational Wave (GW)** time-series data from LIGO using **Deep Convolutional Neural Networks (CNNs)**.
The project processes raw gravitational wave signals from Binary Black Hole System, converts it into spectrograms (2D representation of a signal), and trains a CNN model from the spectrograms data to estimate source parameters, in this case the **mass₁, mass₂, and luminosity distance** of a Binary Black Hole System.

---
## Scientific Background

Gravitational waves are ripples in spacetime produced by accelerating massive objects, such as **Binary Black Hole (BBH)** or **Binary Neutron Star (BNS)** mergers. Since the first direct detection by **LIGO in 2015 (GW150914)**, gravitational-wave astronomy has become an important tool for studying compact objects and testing General Relativity.

Traditional parameter estimation methods in gravitational-wave astronomy rely on **Bayesian inference**, which can be computationally expensive because they require generating millions of waveform templates. This project explores an alternative approach using **Deep Learning**, where a CNN learns the mapping between signal representations and astrophysical source parameters.

---

## Objectives

The main objectives of this project are:

- Understand the characteristics of real gravitational-wave signals from LIGO.
- Generate synthetic BBH signals using **PyCBC**.
- Transform 1D time-series strain data into **2D spectrograms** suitable for CNN processing.
- Train a regression-based CNN model to predict:

  - **Primary black hole mass (mass₁)**
  - **Secondary black hole mass (mass₂)**
  - **Luminosity distance**
- Evaluate the feasibility of deep learning for rapid gravitational-wave parameter estimation.
---

## Project Overview

**Pipeline Summary:**
1. Fetching GW data from [GWTC](https://www.gw-openscience.org/eventapi/html/GWTC/) event API.
2. Extract the signal's time series data and analyze them using **FFT (Fast Fourier Transform)**.
3. Generate simulated noise and synthetic GW signals with PyCBC.
4. Convert the signal's time series data into Spectrograms using `scipy.signal.spectrogram` + **TensorFlow** resizing.
5. Build and train a regression CNN model to estimate GW event properties such as **mass₁, mass₂, and luminosity distance**.
6. Visualize Results: Loss curves, and model architecture.

---

## Dataset

### Real LIGO Data

The project uses publicly available events from the **Gravitational Wave Open Science Center (GWOSC)** catalog.

| Source         | Description                 |
| -------------- | --------------------------- |
| GWTC Catalog   | Real detected GW events     |
| Strain Data    | Time-series detector strain |
| Event Metadata | Published source parameters |

### Simulated Data

Synthetic training samples are generated using **PyCBC** by varying:

| Parameter             | Typical Range |
| --------------------- | ------------- |
| `mass1`               | 5–45 M☉       |
| `mass2`               | 5–45 M☉       |
| `luminosity_distance` | 600–1945 Mpc  |

Noise is added to simulate realistic detector observations.

---

## Signal Processing Workflow

### 1. Time-Series Acquisition

Raw strain data are downloaded from GWOSC and stored as uniformly sampled time-series arrays.

### 2. Frequency Analysis

The signal is transformed using **Fast Fourier Transform (FFT)** to inspect its frequency content:

```python
fft_signal = np.fft.rfft(signal)
frequencies = np.fft.rfftfreq(len(signal), d=dt)
```

This step helps identify the characteristic **chirp pattern** of compact binary mergers.

### 3. Spectrogram Generation

A spectrogram is computed using **Short-Time Fourier Transform (STFT)**:

```python
f, t, Sxx = spectrogram(
    signal,
    fs=sampling_rate,
    nperseg=256,
    noverlap=128
)
```

The resulting spectrogram is resized using TensorFlow to a fixed image dimension for CNN training.

---
## Deep Learning Methodology

### Why Spectrograms?

Although gravitational-wave data are naturally 1D signals, spectrograms provide a **time–frequency representation** that makes chirp evolution visually distinguishable. CNNs are highly effective at learning spatial patterns from such representations.

### Input Pipeline

```text
Raw Strain
    ↓
FFT / Filtering
    ↓
Spectrogram
    ↓
Normalization
    ↓
TensorFlow Dataset
    ↓
CNN Model
```
## CNN Architecture

The model is designed as a **multi-output regression network**.

### Simplified Architecture

```text
Input (128×128×1)
        ↓
Conv2D + ReLU
        ↓
MaxPooling2D
        ↓
Conv2D + ReLU
        ↓
MaxPooling2D
        ↓
Conv2D + ReLU
        ↓
Flatten
        ↓
Dense (256)
        ↓
Dropout
        ↓
Dense (128)
        ↓
Output (mass₁, mass₂, distance)
```

### Example Keras Definition

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(3)
])
```

## Training Configuration

| Hyperparameter   | Value                    |
| ---------------- | ------------------------ |
| Optimizer        | Adam                     |
| Loss Function    | Mean Squared Error (MSE) |
| Batch Size       | 32                       |
| Epochs           | 50–100                   |
| Validation Split | 20%                      |
| Framework        | TensorFlow / Keras       |

---

## Evaluation Metrics

Because this is a **regression problem**, the following metrics are used:

### Mean Absolute Error (MAE)
Primary loss function during training; penalizes larger errors more heavily, useful for guiding gradient descent.

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

### Root Mean Squared Error (RMSE)
Absolute Error tracked alongside MSE as it's more interpretable reported directly in the same physical units as the target (ie. Luminosity Distance, Mpc).

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
$$

### Coefficient of Determination (R²)

Used to measures how well the predicted values explain the variance of the true target values. An **R² value close to 1** indicates that the model explains most of the variance in the data, while a value close to **0** indicates poor explanatory power.

$$
R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}
$$

Where:

* $y_i$ = true value
* $\hat{y}_i$ = predicted value
* $\bar{y}$ = mean of the true values
* $N$ = number of samples

---

## Visualization

The project includes several visual outputs:

### Training History

* Training vs validation loss
* MAE curves across epochs

### Spectrogram Samples

```text
Time ↑
     |
Freq |
     |
     +────────→
```

### Corner Plots

The `GW_Corner/` directory contains **corner plots** comparing:

* **Predicted parameters**
* **Ground-truth simulated values**
* **Published LIGO posterior distributions**

These plots are commonly used in astrophysical parameter-estimation studies.

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| **Core Libraries** | `numpy`, `pandas`, `scipy`, `tqdm`, `requests`, `gzip`, `PyCBC` |
| **Deep Learning** | `TensorFlow`, `Keras`, `sklearn` |
| **Visualization** | `matplotlib`, `plot_model`, `corner` |
| **File I/O** | `pickle`, `os`, `shutil` |

--- 

## Repository Structure
```
GravitationalWave_DeepLearning_project/
├── GW_Corner                           # Corner plots to compare the estimation result with the true result published by LIGO
|
├── GW_catalog/                         # Event catalog of real LIGO data
|   ├── GWTC_123.csv
|
├── implementation_example/             # Usage example of the pipeline using colab jupyter notebook
|   ├── gw_cnn_portfolio_project.ipynb
│                 
├── scripts/                            
│   ├── data_generator/                 # script for prepare the data 
|       ├── real_data.py
|       ├── simulated_data_generator.py
│   ├── data_training/                  # script for training the data
|       ├── data_preprocessing.py
|       ├── GWCNN_trainer.py
│
├── README.md                           # Project overview and instructions
├── LICENSE                             # License information for the repository
```

## License

This project is licensed under the [MIT License]({{ '/license/' | relative_url }}). You are free to use, modify, and share this project with proper attribution.

## About Me

Hi there! I'm **Sadam Habnawi**. I'm a physics fresh graduate, i have a great enthusiasm in the field of data including data analytics, engineering, and data science!

Let's stay in touch! Feel free to connect with me on the following platforms:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/sadam-habnawi-7621011b4)

---

## Acknowledgements

- [LIGO Scientific Collaboration](https://www.ligo.org)
- [PyCBC Project](https://github.com/gwastro/pycbc)
- [Keras & TensorFlow](https://www.tensorflow.org)
- [GWTC Public Data Release](https://www.gw-openscience.org/eventapi/html/GWTC/)
