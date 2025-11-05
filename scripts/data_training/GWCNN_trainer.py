# ======================================================================================================================
# This code is the pipeline to train the spectrograms of simulated waveform GW data
#
# Usage Example :
# from data_preprocessing import WaveformDatasetLoader
# trainer = SpectrogramCNNTrainer(
#   sim_data_dir="GW_sim_data/training_data",
#   real_data_dir="GW_real_data/waveform/training_data",
#   model_output_dir="model_output")
# trainer.run_pipeline()
# ======================================================================================================================

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, backend as K
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class SpectrogramCNNTrainer:
    def __init__(self,
                 sim_data_dir,
                 real_data_dir,
                 model_output_dir,
                 test_size=0.2,
                 random_state=42):
        """
        Handles loading, normalizing, training, and saving a CNN for spectrogram data.

        Args:
            sim_data_dir (str): Path to folder containing simulation spectrogram .npy/.txt files.
            real_data_dir (str): Path to folder containing real spectrogram .npy/.txt files.
            model_output_dir (str): Directory to save trained model and history.
            test_size (float): Fraction for test data split.
            random_state (int): Random seed for reproducibility.
        """
        self.sim_data_dir = sim_data_dir
        self.real_data_dir = real_data_dir
        self.model_output_dir = model_output_dir
        self.test_size = test_size
        self.random_state = random_state
        os.makedirs(model_output_dir, exist_ok=True)

        self.X_sim, self.y_sim = None, None
        self.X_real, self.y_real = None, None
        self.model = None
        self.history = None

    # ----------------------------------------------------------
    def load_data(self):
        """Load simulated and real spectrogram data."""
        tqdm.write("Loading spectrogram datasets...")

        self.X_sim = np.load(os.path.join(self.sim_data_dir, "X_spectrograms.npy"))
        self.y_sim = np.load(os.path.join(self.sim_data_dir, "y_labels.npy"))

        self.X_real = np.load(os.path.join(self.real_data_dir, "X_spectrograms.npy"))
        self.y_real = np.load(os.path.join(self.real_data_dir, "y_labels.npy"))

        tqdm.write(f"Loaded simulation data: {self.X_sim.shape}")
        tqdm.write(f"Loaded real data: {self.X_real.shape}\n")

    # ----------------------------------------------------------
    def normalize_targets(self):
        """Normalize target labels (mass1, mass2, distance) to 0–1 range per sample."""
        def normalize_y(y):
            for i in range(len(y)):
                y[i, 0] /= np.max(y[i, 0])
                y[i, 1] /= np.max(y[i, 1])
                y[i, 2] /= np.max(y[i, 2])
            return y

        tqdm.write("⚙️ Normalizing y values...")
        self.y_sim = normalize_y(self.y_sim)
        self.y_real = normalize_y(self.y_real)
        tqdm.write("Normalization complete.\n")

    # ----------------------------------------------------------
    @staticmethod
    def R2(y_true, y_pred):
        """Custom R² metric."""
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    # ----------------------------------------------------------
    @staticmethod
    def build_cnn_model(input_shape):
        """Build a CNN model for spectrogram regression."""
        K.clear_session()
        model = Sequential([

            layers.Input(input_shape),

            layers.Conv2D(128, (6,6), padding='same', kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(64, (6,6), padding='same', kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(32, (6,6), padding='same', kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(64, (6,6), padding='same', kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(128, (6,6), padding='same', kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),

            layers.Conv2D(16, (6,6), padding='same', kernel_initializer='glorot_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2,2)),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(64),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(3)
        ])
        return model

    # ----------------------------------------------------------
    def train_cnn_model(self, X_train, y_train):
        """Compile, train, and validate CNN model."""
        tqdm.write("Starting CNN training...")
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        cnn_model = self.build_cnn_model(input_shape)

        cnn_model.compile(
            optimizer=optimizers.Adam(),
            loss='Huber',
            metrics=[self.R2]
        )

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        lr_schedule = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=15,
            min_lr=1e-12
        )

        history = cnn_model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )

        tqdm.write("CNN training complete.\n")
        self.model = cnn_model
        self.history = history
        return cnn_model, history

    # ----------------------------------------------------------
    def save_model_and_history(self, model_name="model_gwcnn"):
        """Save trained model (.keras) and training history (.pkl)."""
        model_path = os.path.join(self.model_output_dir, f"{model_name}.keras")
        hist_path = os.path.join(self.model_output_dir, f"history_{model_name}.pkl")

        self.model.save(model_path)
        with open(hist_path, "wb") as f:
            pickle.dump(self.history.history, f)

        tqdm.write(f"Saved model: {model_path}")
        tqdm.write(f"Saved training history: {hist_path}")

    # ----------------------------------------------------------
    def run_pipeline(self):
        """Full pipeline: load -> normalize -> split -> train -> save."""
        tqdm.write("Starting Spectrogram CNN training pipeline...\n")

        # 1. Load data
        self.load_data()

        # 2. Normalize labels
        self.normalize_targets()

        # 3. Train-test split (only on simulated data)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_sim, self.y_sim,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # 4. Train model
        self.train_cnn_model(X_train, y_train)

        # 5. Save results
        self.save_model_and_history()

        tqdm.write("Training pipeline complete!\n")
