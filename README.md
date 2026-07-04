# Gravitational Wave with Deep Learning Project

Welcome to the **Gravitational Wave with Deep Learning Project** repository!  
This repository is my final project for my study in pyhsics. This repository contains a complete Deep Learning pipeline for analyzing **gravitational wave (GW)** time-series data from LIGO using **deep convolutional neural networks (CNNs)**.
The project processes raw strain data, converts it into spectrograms, and trains a CNN model to estimate source parameters, in this case the **mass₁, mass₂, and luminosity distance** of a Binary Black Hole System.

---

## Project Overview

**Pipeline Summary:**
1. **Download GW Data** from [GWTC Catalog](https://www.gw-openscience.org/eventapi/html/GWTC/).
2. **Extract Strain Files** and preprocess them.
3. **Generate Simulated Noise** and synthetic GW signals using PyCBC.
4. **Convert Time-Series → Spectrograms** using `scipy.signal.spectrogram` + TensorFlow resizing.
5. **Train a CNN** to predict GW event properties.
6. **Visualize Results**: Loss curves, and model architecture.

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| **Core Libraries** | `numpy`, `pandas`, `scipy`, `tqdm`, `requests`, `gzip`, `PyCBC` |
| **Deep Learning** | `TensorFlow`, `Keras`, `sklearn` |
| **Visualization** | `matplotlib`, `plot_model` |
| **File I/O** | `pickle`, `os`, `shutil` |

--- 

## Repository Structure
```
GravitationalWave_DeepLearning_project/
│
├── Example/                            # Usage example of the pipeline using colab jupyter notebook 
│
├── GW_catalog/                         # Event catalog of real LIGO data
│                 
├── scripts/                            
│   ├── data_generator/                 # script for prepare the data 
│   ├── data_training/                  # script for training the data
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
