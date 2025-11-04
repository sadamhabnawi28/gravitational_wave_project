# ======================================================================================================================
# This is the pipeline to genarate gravitational waves simulations data (GW waveforms + simulated transient noise) 
# To run this pipeline 
# first, generate the noise samples with specified real data folder and the output folder
# second, run the simulation data generator with specified mass range and distance range 
#
# Example :
# from GW_simulation_data_generator import L1NoiseGenerator, GWTCWaveformSimulator
# noise_gen = L1NoiseGenerator(
#     data_folder=".../gravitational_wave_real_data",
#     output_folder=".../generated_noises")
# generated = noise_gen.generate_all_noises()
# print(generated)
# sim = GWTCWaveformSimulator(
#     noise_folder="/content/drive/MyDrive/Colab Notebooks/GW_CNN_portfolio_project/generated_noises", 
#     output_folder="/content/drive/MyDrive/Colab Notebooks/GW_CNN_portfolio_project/GW_L1_sim_data")
# mass_range = np.linspace(5, 45, 8)
# distance_range = np.linspace(600, 1945, 4)
# sim.run_simulations(mass_range, distance_range)
# ======================================================================================================================

import os
import numpy as np
from tqdm import tqdm
from pycbc.noise.reproduceable import colored_noise
from pycbc.types import TimeSeries
from pycbc.psd import (
    aLIGOZeroDetHighPower, aLIGOZeroDetLowPower, aLIGOaLIGO175MpcT1800545,
    aLIGOThermal, AdvVirgo, AdVO3LowT1800545, AdVEarlyLowSensitivityP1200087,
    AdVEarlyHighSensitivityP1200087, AdVBNSOptimizedSensitivityP1200087,
    AdVDesignSensitivityP1200087)

# Noise Samples Generator
class L1NoiseGenerator:
    def __init__(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def _generate_noise(self, input_filename, psd_func):
        """Generate a single noise sample for an event file."""
        event_id = os.path.splitext(input_filename)[0].split('_')[1]
        save_name = f"L1_{event_id}_noise.txt"
        save_path = os.path.join(self.output_folder, save_name)

        strain_data = np.loadtxt(os.path.join(self.data_folder, input_filename), skiprows=3)
        strain = TimeSeries(strain_data, 1/4096)
        delta_f = strain.delta_f
        psd = psd_func(delta_f)

        noise = colored_noise(psd, 0, 32, seed=123, sample_rate=4096)
        np.savetxt(save_path, noise, header=f"Noise sample for {event_id}")
        return save_path

    def generate_all_noises(self, overwrite=False):
        """Generate 4 L1 noise files with tqdm progress bar."""
        configs = [
            ("L1_GW151012-v3.txt", self.psd_15),
            ("L1_GW170809-v1.txt", self.psd_17),
            ("L1_GW190517_055101-v1.txt", self.psd_19),
            ("L1_GW200225_060421-v1.txt", self.psd_20)
        ]

        generated_files = []
        for filename, psd_func in tqdm(configs, desc="🎧 Generating L1 noise samples", unit="file"):
            event_id = os.path.splitext(filename)[0].split('_')[1]
            save_path = os.path.join(self.output_folder, f"L1_{event_id}_noise.txt")
            if not os.path.exists(save_path) or overwrite:
                generated_files.append(self._generate_noise(filename, psd_func))
            else:
                generated_files.append(save_path)
        return generated_files

    # PSD generator functions for each noise
    def psd_15(self, delta_f):
        psd = AdVBNSOptimizedSensitivityP1200087(32*4096, delta_f, 15)/43
        psd += aLIGOZeroDetHighPower(32*4096, delta_f, 15)*5
        psd += AdVO3LowT1800545(32*4096, delta_f, 15)/20
        psd += aLIGOaLIGO175MpcT1800545(32*4096, delta_f, 15)/200
        psd += aLIGOThermal(32*4096, delta_f, 15)/80
        psd += AdvVirgo(32*4096, delta_f, 15)/700
        psd += AdVDesignSensitivityP1200087(32*4096, delta_f, 15)/1000
        psd += AdVEarlyHighSensitivityP1200087(32*4096, delta_f, 15)/5e3
        psd += AdVEarlyLowSensitivityP1200087(32*4096, delta_f, 15)/37.5
        return psd

    def psd_17(self, delta_f):
        psd = AdVBNSOptimizedSensitivityP1200087(32*4096, delta_f, 15)/10.5
        psd += aLIGOZeroDetLowPower(32*4096, delta_f, 15)
        psd += AdVDesignSensitivityP1200087(32*4096, delta_f, 15)/6.22
        psd += AdVEarlyHighSensitivityP1200087(32*4096, delta_f, 15)/450
        psd += AdVEarlyLowSensitivityP1200087(32*4096, delta_f, 15)/600
        return psd

    def psd_19(self, delta_f):
        psd = aLIGOaLIGO175MpcT1800545(32*4096, delta_f, 15)
        psd += AdVBNSOptimizedSensitivityP1200087(32*4096, delta_f, 15)/800
        psd += aLIGOZeroDetHighPower(32*4096, delta_f, 15)/1.51
        psd += AdVO3LowT1800545(32*4096, delta_f, 15)/100
        psd += aLIGOThermal(32*4096, delta_f, 15)/1000
        psd += AdvVirgo(32*4096, delta_f, 15)/1000
        return psd

    def psd_20(self, delta_f):
        psd = AdVBNSOptimizedSensitivityP1200087(32*4096, delta_f, 15)/50
        psd += AdVO3LowT1800545(32*4096, delta_f, 15)/40
        psd += aLIGOaLIGO175MpcT1800545(32*4096, delta_f, 15)/65
        psd += AdVDesignSensitivityP1200087(32*4096, delta_f, 15)/1.319
        return psd

# GW Simulations Generator
class GWTCWaveformSimulator:
    def __init__(self, noise_folder, output_folder):
        self.noise_folder = noise_folder
        self.output_folder = output_folder
        self.sample_rate = 4096
        self.f_lower = 30
        self.total_duration = 32
        self.start_time = 0
        os.makedirs(self.output_folder, exist_ok=True)
        # Load noise files
        self.noise_files = [np.loadtxt(os.path.join(noise_folder, f"L1_{n}_noise.txt"), skiprows=1)
                            for n in ["GW151012-v3", "GW170809-v1", 
                                      "GW190517", "GW200225"]]
    def add_noise(self, hp):
        """Insert waveform into random noise background."""
        total_samples = self.total_duration * self.sample_rate
        waveform_data = np.zeros(total_samples)
        waveform_data[15*self.sample_rate:15*self.sample_rate+len(hp)] = hp.data
        waveform_data += random.choice(self.noise_files)
        return waveform_data

    def get_waveform(self, strain, merge_time):
        """Whiten and filter waveform data."""
        strain = TimeSeries(strain, delta_t=1/self.sample_rate, epoch=self.start_time)
        strain = highpass(strain, 30).crop(2, 2)
        psd = interpolate(strain.psd(4), strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4*strain.sample_rate), low_frequency_cutoff=self.f_lower)
        white_data = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
        white_data = white_data.highpass_fir(30, 512).lowpass_fir(300, 512)
        return white_data.time_slice(merge_time-0.8, merge_time+0.2)

    def run_simulations(self, mass_range, distance_range):
        """Run all waveform simulations with tqdm progress bars."""
        sim_count = 0
        total_jobs = sum(1 for m1 in mass_range for m2 in mass_range if m2 >= m1 for _ in distance_range)

        with tqdm(total=total_jobs, desc="Generating waveforms", unit="sim") as pbar:
            for mass1 in mass_range:
                for mass2 in mass_range:
                    if mass2 >= mass1:
                        for distance in distance_range:
                            hp, _ = get_td_waveform(
                                approximant="SEOBNRv4_opt",
                                mass1=mass1, mass2=mass2, distance=distance,
                                delta_t=1/self.sample_rate, f_lower=self.f_lower)
                            noisy_strain = self.add_noise(hp)
                            merge_time = 15 + len(hp)/self.sample_rate
                            processed_wave = self.get_waveform(noisy_strain, merge_time)
                            filename = f"Waveform_Mass1_{mass1:.1f}_Mass2_{mass2:.1f}_Distance_{distance:.1f}.txt"
                            np.savetxt(os.path.join(self.output_folder, filename), processed_wave, header="Noisy Strain h+")
                            sim_count += 1
                            pbar.update(1)
        tqdm.write(f"{sim_count} gravitational wave simulations completed and saved.")



