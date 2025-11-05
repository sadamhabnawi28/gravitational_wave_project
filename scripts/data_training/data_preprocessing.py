# ===========================================================================================================================
# This code is the pipeline to preprocess the simulated data before training. This pipeline consist of the data loading 
# from waveforms dierectory into a single .txt file and the conversion into spectrogram for each waveform from that txt file.
#
# Example :
# from data_preprocessing import WaveformDatasetLoader
# loader = WaveformDatasetLoader(data_dir="GW_sim_data")
# Run the full preprocessing pipeline
# loader.run_pipeline()
# X_specs, y_labels = loader.spectrograms, loader.labels
# print("Spectrograms shape:", X_specs.shape)
# print("Labels shape:", y_labels.shape)
# =============================================================================================================================

import os
import numpy as np
from tqdm import tqdm
from scipy import signal
import tensorflow as tf

class WaveformDatasetLoader:
    def __init__(self, data_dir, output_subdir="training_data"):
        """
        Handles loading, saving, and converting waveform simulation data.

        Args:
            data_dir (str): Directory containing waveform .txt files.
            output_subdir (str): Folder inside data_dir to store saved numpy arrays.
        """
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.X = None
        self.y = None
        self.spectrograms = None
        self.labels = None
        
    def _parse_filename(self, filename):
        """
        Extracts mass1, mass2, and distance values from a standardized filename.
        Expected format: waveform_mass1_<m1>_mass2_<m2>_distance_<dist>_<det>_<event>.txt
        """
        parts = filename.split('_')
        try:
            mass1 = float(parts[2])
            mass2 = float(parts[4])
            distance = float(parts[6].replace(".txt", ""))
            return mass1, mass2, distance
        except (IndexError, ValueError):
            raise ValueError(f"Filename '{filename}' does not follow expected format.")

    def load_data(self):
        """Load waveform .txt files into numpy arrays (time series)."""
        X, y = [], []
        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]

        for filename in tqdm(txt_files, desc="Loading waveform data", unit="file"):
            filepath = os.path.join(self.data_dir, filename)
            try:
                mass1, mass2, distance = self._parse_filename(filename)
                strain = np.loadtxt(filepath, skiprows=1)
                X.append(strain)
                y.append([mass1, mass2, distance])
            except Exception as e:
                tqdm.write(f"Skipping {filename}: {e}")

        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)
        tqdm.write(f"Loaded {len(self.X)} waveform samples.\n")
        return self.X, self.y

    # ------------------------------------------------------------------
    def save_arrays(self):
        """Save X and y arrays to the output directory."""
        if self.X is None or self.y is None:
            raise RuntimeError("Data not loaded yet. Run load_data() first.")
        X_path = os.path.join(self.output_dir, "X_sim.txt")
        y_path = os.path.join(self.output_dir, "y_sim.txt")
        np.savetxt(X_path, self.X, delimiter=",")
        np.savetxt(y_path, self.y, delimiter=",")
        tqdm.write(f"Saved arrays to:\n- {X_path}\n- {y_path}")

    def convert_to_spectrograms(self, resize_shape=(128, 128), fs=4096):
        """
        Convert loaded waveform time-series data into 2D spectrograms.

        Args:
            resize_shape (tuple): Desired spectrogram image size (H, W).
            fs (int): Sampling rate in Hz.

        Returns:
            tuple: (spectrograms, labels)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Data not loaded yet. Run load_data() first.")

        spectrograms = []
        labels = []

        for i in tqdm(range(len(self.X)), desc="Converting to spectrograms", unit="waveform"):
            strain = self.X[i]
            label = self.y[i]

            freq, times, spec = signal.spectrogram(
                strain,
                fs=fs,
                nperseg=512,
                noverlap=256
            )

            # Normalize and crop
            spec /= spec.max() + 1e-12
            freq_mask = freq <= 512
            spec_cropped = spec[freq_mask, :]

            # Expand and resize
            spec_cropped = np.expand_dims(spec_cropped, axis=-1)
            spec_resized = tf.image.resize(spec_cropped, size=resize_shape).numpy()

            spectrograms.append(spec_resized)
            labels.append(label)

        self.spectrograms = np.array(spectrograms, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)

        tqdm.write(f"Generated {len(self.spectrograms)} spectrograms ({resize_shape[0]}×{resize_shape[1]}).")
        return self.spectrograms, self.labels

    def save_spectrograms(self):
        """Save spectrograms and labels as .npy files."""
        if self.spectrograms is None or self.labels is None:
            raise RuntimeError("Spectrograms not generated yet. Run convert_to_spectrograms() first.")

        X_path = os.path.join(self.output_dir, "X_spectrograms.npy")
        y_path = os.path.join(self.output_dir, "y_labels.npy")
        np.save(X_path, self.spectrograms)
        np.save(y_path, self.labels)
        tqdm.write(f"Saved spectrogram data:\n- {X_path}\n- {y_path}")

    def run_pipeline(self):
        """Full pipeline: load -> convert -> save."""
        tqdm.write("Running full waveform-to-spectrogram pipeline...\n")
        self.load_data()
        self.convert_to_spectrograms()
        self.save_spectrograms()
        tqdm.write("Pipeline complete!\n")

