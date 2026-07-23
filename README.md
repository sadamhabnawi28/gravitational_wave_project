# Gravitational Wave with Deep Learning Project

Welcome to the **Gravitational Wave with Deep Learning Project** repository!  
This repository was my final project for my study in pyhsics. This repository contains a complete Deep Learning pipeline for analyzing **Gravitational Wave (GW)** time-series data from LIGO using **Deep Convolutional Neural Networks (CNNs)**.
The project processes raw gravitational wave signals from Binary Black Hole System, converts it into spectrograms (2D representation of a signal), and trains a CNN model from the spectrograms data to estimate source parameters, in this case the **mass₁, mass₂, and luminosity distance** of a Binary Black Hole System.

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
---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.

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
