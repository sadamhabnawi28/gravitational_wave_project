# ==============================================================================================================
# This code is the pipeline to get the real LIGO GW data
# To run this pipeline specify the catalog file and the output directory into the GWTCDataHandler class
# Example :
# from GW_real_data import GWTCDataHandler
# gwtc_handler = GWTCDataHandler(
#     gwtc_csv_path="GWTC_123.csv",
#     output_folder="gravitational_wave_real_data",
#     waveform_output="gravitational_wave_real_data/waveform",
#     max_workers=10)
# gwtc_handler.run_pipeline()
# ==============================================================================================================

import os
import io
import gzip
import json
import shutil
import random
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pycbc.types import TimeSeries
from pycbc.filter import highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform


class GWTCDataHandler:
    def __init__(self, gwtc_csv_path, output_folder,
                 waveform_output, max_workers=8):
        """
        Full GWTC data handler: download -> extract -> process waveform.
        """
        self.gwtc_csv_path = gwtc_csv_path
        self.output_folder = output_folder
        self.waveform_output = waveform_output
        self.max_workers = max_workers
        
        self.gwtc_df = None
        self.json_data = []
        self.strain_metadata = []
        self.url_list = []

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.waveform_output, exist_ok=True)

    def load_gwtc_catalog(self):
        """Load the GWTC catalog CSV file."""
        print("Loading GWTC catalog...")
        self.gwtc_df = pd.read_csv(self.gwtc_csv_path)
        print(f"Loaded {len(self.gwtc_df)} catalog entries.\n")

    def fetch_json_data(self):
        """Fetch JSON data from each event URL in the catalog."""
        print("Fetching JSON metadata from URLs...")
        urls = self.gwtc_df['jsonurl'].tolist()
        for url in tqdm(urls, desc="Downloading JSON metadata", unit="file"):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.json_data.append(response.json())
                else:
                    tqdm.write(f"Failed: {url} (HTTP {response.status_code})")
            except requests.RequestException as e:
                tqdm.write(f"Error fetching {url}: {e}")
        tqdm.write(f"Total JSON files fetched: {len(self.json_data)}\n")

    def extract_strain_metadata(self):
        """Extract strain metadata for H1 or L1 detectors."""
        print("Extracting strain metadata...")
        strain_data = []
        for event_data in tqdm(self.json_data, desc="Processing JSON files", unit="file"):
            for event_name, event_info in event_data['events'].items():
                for strain in event_info['strain']:
                    if (strain['detector'] in ['H1', 'L1'] and 
                        strain['sampling_rate'] == 4096 and 
                        strain['duration'] == 32 and 
                        strain['format'] == 'txt'):
                        strain_data.append({
                            'name': event_name,
                            'detector': strain['detector'],
                            'url': strain['url']
                        })
        self.strain_metadata = strain_data
        tqdm.write(f"Extracted {len(strain_data)} valid strain entries.\n")

    def generate_url_list(self):
        """Generate list of URLs with detector and event name."""
        print("Building URL list...")
        df_meta = pd.DataFrame(self.strain_metadata)
        df_merged = df_meta.merge(self.gwtc_df[['commonName', 'jsonurl']],
                                  left_on='name', right_on='jsonurl', how='left')
        df_merged['commonName'] = df_merged['commonName'].fillna(df_merged['name'])

        self.url_list = list(zip(df_merged['commonName'], df_merged['detector'], df_merged['url']))
        tqdm.write(f"URL list built with {len(self.url_list)} entries.\n")
        return df_merged[['commonName', 'detector', 'url']]

    def _download_and_extract_single(self, url_info):
        """Download and extract a single strain file."""
        common_name, detector, url = url_info
        file_name = f"{detector}_{common_name}.txt"
        save_path = os.path.join(self.output_folder, file_name)

        if os.path.exists(save_path):  
            return f"Skipped (exists): {file_name}"

        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                compressed_stream = io.BytesIO(response.content)
                with gzip.open(compressed_stream, 'rb') as f_in, open(save_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return f"Saved: {file_name}"
            else:
                return f"Failed ({response.status_code}): {file_name}"
        except requests.RequestException as e:
            return f"Error {file_name}: {e}"

    def download_and_extract_parallel(self):
        """Download and extract all strain data."""
        total_files = len(self.url_list)
        print(f"Downloading and extracting {total_files} strain files...\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_and_extract_single, info)
                       for info in self.url_list]
            with tqdm(total=total_files, desc="Extracting strain files", unit="file") as pbar:
                for future in as_completed(futures):
                    _ = future.result()
                    pbar.update(1)

        tqdm.write("All strain data downloaded and extracted!\n")

    def _get_waveform(self, strain, start_time, sample_rate, merge_time):
        """Whiten and filter the strain waveform."""
        strain = TimeSeries(strain, delta_t=1/sample_rate, epoch=start_time)
        strain = highpass(strain, 30)
        strain = strain.crop(2, 2)
        psd = strain.psd(4)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=30)
        white_data = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
        white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
        return white_data.time_slice(merge_time - 0.8, merge_time + 0.2)

    def process_waveforms(self):
        """Process and whiten all downloaded strain files."""
        print("Processing downloaded gravitational wave data...\n")
        df = self.gwtc_df
        rand_merge = random.choice([-0.05, 0, 0.1, 0.2, 0.3, 0.4])
        count = 0

        data_files = [f for f in os.listdir(self.output_folder) if f.endswith(".txt")]

        for filename in tqdm(data_files, desc="Generating whitened waveforms", unit="file"):
            filepath = os.path.join(self.output_folder, filename)
            det, common_name = filename.split("_", 1)
            common_name = common_name.replace(".txt", "")
            try:
                row = df[(df["id"] == common_name) & (df["detector"] == det)].iloc[0]
            except IndexError:
                tqdm.write(f"No metadata found for {filename}")
                continue

            strain = np.loadtxt(filepath)
            mass1, mass2, distance = row["mass_1_source"], row["mass_2_source"], row["luminosity_distance"]
            start_time, gps_time = row["start_GPS"], row["GPS"]

            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=mass1, mass2=mass2, distance=distance,
                                    delta_t=1/4096, f_lower=30)
            waveform = self._get_waveform(strain, start_time, 4096, gps_time + rand_merge)
            outname = f"waveform_mass1_{mass1:.1f}_mass2_{mass2:.1f}_distance_{distance:.1f}_{det}_{common_name}.txt"
            outpath = os.path.join(self.waveform_output, outname)
            np.savetxt(outpath, waveform, header="Real Data")
            count += 1

        tqdm.write(f"{count} gravitational wave real data waveforms processed and saved.")
        tqdm.write("DONE\n")

    def run_pipeline(self):
        """Run the full GWTC data pipeline: download → extract → process."""
        print("Starting GWTC data pipeline...\n")
        self.load_gwtc_catalog()
        self.fetch_json_data()
        self.extract_strain_metadata()
        self.generate_url_list()
        self.download_and_extract_parallel()
        self.process_waveforms()
        print("Pipeline completed successfully!")




