### **Mount Google drive**

from google.colab import drive
drive.mount('/content/drive')

### **Install required libraries**

!pip -q install neurokit2

!pip -q install ts2vg

### **Calculating HRV

**1.**


import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def process_ecg_file(file_path, start=None, end=None, chunk_size=15000):
    read_file = pd.read_csv(file_path)
    if 'time' not in read_file.columns or read_file['time'].dropna().empty:
        print(f"Skipping file {file_path} as 'time' column has no data.")
        return

    if start is not None and end is not None:
        read_file = read_file.iloc[start:end]

    read_file['time'] = pd.to_datetime(read_file['time'])
    total_time = (read_file['time'].max() - read_file['time'].min()).total_seconds()
    print(f"Total time for {file_path}: {total_time:.2f} seconds")

    n = len(read_file) // total_time
    sampling = int(n)
    print(f"Sampling rate for {file_path}: {sampling}")

    num_chunks = len(read_file) // chunk_size
    all_rpeaks = []

    file_name = os.path.basename(file_path)
    if '101_ECG' <= file_name <= '175_ECG':
        column_index = 2
    else:
        column_index = 1

    for i in range(num_chunks + 1):
        start_row = i * chunk_size
        end_row = (i + 1) * chunk_size
        ecg_signal = read_file.iloc[start_row:end_row, column_index].values

        if len(ecg_signal) == 0:
            continue

        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling, method='vg')

        lowcut = 4.0
        highcut = 50.0
        fs = sampling

        def band_pass_filter(ecg_cleaned, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(order, [low, high], btype='band')
            padlen = min(len(ecg_cleaned) - 1, int(0.25 * len(ecg_cleaned)))
            y = signal.filtfilt(b, a, ecg_cleaned, padlen=padlen)
            return y

        ecg_filtered = band_pass_filter(ecg_cleaned, lowcut, highcut, fs)
        _, emrich2023 = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling, method="emrich2023")

        adjusted_rpeaks = {key: value + start_row for key, value in emrich2023.items() if key == "ECG_R_Peaks"}
        all_rpeaks.append(adjusted_rpeaks)

        plt.figure(figsize=(15, 6))
        x_values = np.arange(start_row, start_row + len(ecg_filtered))
        plt.plot(x_values, ecg_filtered, label='ECG Signal')

        r_peaks = emrich2023['ECG_R_Peaks']
        plt.plot(x_values[r_peaks], ecg_filtered[r_peaks], 'ro', markersize=5, label='R Peaks')

        plt.title(f'ECG Signal with R Peaks for {os.path.basename(file_path)} - Chunk {i+1}-{column_index}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    combined_rpeaks = np.concatenate([r["ECG_R_Peaks"] for r in all_rpeaks if "ECG_R_Peaks" in r])
    print(f"Combined R-peaks for {file_path}: {combined_rpeaks}")
    n=len(combined_rpeaks)
    total=n/total_time
    avg_hr=total*60
    print(avg_hr)

    if combined_rpeaks.size > 0:
        hrv_results = nk.hrv(combined_rpeaks, sampling_rate=sampling, show=False)
        hrv_results['File_Name'] = file_name
        csv_file = 'combined_hrv.csv'
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file)
            updated_data = pd.concat([existing_data, hrv_results], ignore_index=True)
            updated_data.to_csv(csv_file, index=False)
        else:
            hrv_results.to_csv(csv_file, index=False)
        print(hrv_results)

    return combined_rpeaks

file_ranges = {
    "103_ECG.csv": (0, 129000),
    "105_ECG.csv": (0, 88000),
    "106_ECG.csv": (45000,153620),
    "115_ECG.csv": (94000, 170813),
    "118_ECG.csv": (0, 140000),
    "120_ECG.csv": (3000, 177346),
    "141_ECG.csv": (78000, 162000),
    "143_ECG.csv": (0, 80000),
    "144_ECG.csv": (0, 105000),
    "147_ECG.csv": (0, 195000),
    "148_ECG.csv": (0, 105000),
    "157_ECG.csv": (0, 150000),
    "160_ECG.csv": (52000, 111827),
    "161_ECG.csv": (0, 165000),
    "164_ECG.csv": (0, 75000),
    "169_ECG.csv": (56000, 120000),
    "171_ECG.csv": (12000, 131583),
    "181_ECG.csv": (0, 64000),
    "184_ECG.csv": (0, 225000),
    "194_ECG.csv": (0, 105000),
    "204_ECG.csv": (0, 74000),
    "209_ECG.csv": (0, 75000),
    "214_ECG.csv": (45000, 190736)
}

folder_path = '/content/drive/MyDrive/new_folder'
all_hrv_results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        if filename in file_ranges:
            start, end = file_ranges[filename]
            print(f"Processing file: {file_path} with range {start} to {end}")
            process_ecg_file(file_path, start, end)
        else:
            print(f"Processing file: {file_path} with full range")
            process_ecg_file(file_path)

        if hrv_result is not None:
            all_hrv_results.append(hrv_result)

if all_hrv_results:
    combined_hrv_data = pd.concat(all_hrv_results, ignore_index=True)
    combined_hrv_data.to_csv('combined_hrv_data.csv', index=False)

    # Create box plots for each HRV feature
    hrv_features = combined_hrv_data.columns.difference(['File_Name'])
    for feature in hrv_features:
        plt.figure(figsize=(10, 6))
        combined_hrv_data.boxplot(column=feature)
        plt.title(f'Box Plot for {feature}')
        plt.ylabel('Values')
        plt.show()

r_r_intervals = np.diff(combined_rpeaks) / sampling
heart_rate = 60 / r_r_intervals
heart_rate_df = pd.DataFrame({
    "R_Peaks": r_peaks[1:],
    "R-R_Interval (s)": r_r_intervals,
    "Heart Rate (BPM)": heart_rate
})
print(heart_rate_df)

import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def process_ecg_file(file_path, duration=60, chunk_size=15000):
    read_file = pd.read_csv(file_path)
    if 'time' not in read_file.columns or read_file['time'].dropna().empty:
        print(f"Skipping file {file_path} as 'time' column has no data.")
        return

    read_file['time'] = pd.to_datetime(read_file['time'])
    start_time = read_file['time'].min()
    end_time = start_time + pd.Timedelta(seconds=duration)
    read_file = read_file[(read_file['time'] >= start_time) & (read_file['time'] <= end_time)]

    total_time = (read_file['time'].max() - read_file['time'].min()).total_seconds()
    print(f"Total time for {file_path}: {total_time:.2f} seconds")

    n = len(read_file) // total_time
    sampling = int(n)
    print(f"Sampling rate for {file_path}: {sampling}")

    num_chunks = len(read_file) // chunk_size
    all_rpeaks = []

    file_name = os.path.basename(file_path)
    if '101_ECG' <= file_name <= '175_ECG':
        column_index = 2
    else:
        column_index = 1

    for i in range(num_chunks + 1):
        start_row = i * chunk_size
        end_row = (i + 1) * chunk_size
        ecg_signal = read_file.iloc[start_row:end_row, column_index].values

        if len(ecg_signal) == 0:
            continue

        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling, method='vg')

        lowcut = 4.0
        highcut = 50.0
        fs = sampling

        def band_pass_filter(ecg_cleaned, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(order, [low, high], btype='band')
            padlen = min(len(ecg_cleaned) - 1, int(0.25 * len(ecg_cleaned)))
            y = signal.filtfilt(b, a, ecg_cleaned, padlen=padlen)
            return y

        ecg_filtered = band_pass_filter(ecg_cleaned, lowcut, highcut, fs)
        _, emrich2023 = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling, method="emrich2023")

        adjusted_rpeaks = {key: value + start_row for key, value in emrich2023.items() if key == "ECG_R_Peaks"}
        all_rpeaks.append(adjusted_rpeaks)

        plt.figure(figsize=(15, 6))
        x_values = np.arange(start_row, start_row + len(ecg_filtered))
        plt.plot(x_values, ecg_filtered, label='ECG Signal')

        r_peaks = emrich2023['ECG_R_Peaks']
        plt.plot(x_values[r_peaks], ecg_filtered[r_peaks], 'ro', markersize=5, label='R Peaks')

        plt.title(f'ECG Signal with R Peaks for {os.path.basename(file_path)} - Chunk {i+1}-{column_index}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    combined_rpeaks = np.concatenate([r["ECG_R_Peaks"] for r in all_rpeaks if "ECG_R_Peaks" in r])
    print(f"Total R-peaks for {file_path} in first {duration} seconds: {len(combined_rpeaks)}")

    return combined_rpeaks

folder_path = '/content/drive/MyDrive/new_folder'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path} for the first 60 seconds")
        process_ecg_file(file_path)

import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def process_ecg_file(file_path, start=None, end=None, chunk_size=15000):
    read_file = pd.read_csv(file_path)
    if 'time' not in read_file.columns or read_file['time'].dropna().empty:
        print(f"Skipping file {file_path} as 'time' column has no data.")
        return None

    if start is not None and end is not None:
        read_file = read_file.iloc[start:end]

    read_file['time'] = pd.to_datetime(read_file['time'])
    total_time = (read_file['time'].max() - read_file['time'].min()).total_seconds()
    print(f"Total time for {file_path}: {total_time:.2f} seconds")

    n = len(read_file) // total_time
    sampling = int(n)
    print(f"Sampling rate for {file_path}: {sampling}")

    num_chunks = len(read_file) // chunk_size
    all_rpeaks = []

    file_name = os.path.basename(file_path)
    if '101_ECG' <= file_name <= '175_ECG':
        column_index = 2
    else:
        column_index = 1

    for i in range(num_chunks + 1):
        start_row = i * chunk_size
        end_row = (i + 1) * chunk_size
        ecg_signal = read_file.iloc[start_row:end_row, column_index].values

        if len(ecg_signal) == 0:
            continue

        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling, method='vg')

        lowcut = 4.0
        highcut = 50.0
        fs = sampling

        def band_pass_filter(ecg_cleaned, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(order, [low, high], btype='band')
            padlen = min(len(ecg_cleaned) - 1, int(0.25 * len(ecg_cleaned)))
            y = signal.filtfilt(b, a, ecg_cleaned, padlen=padlen)
            return y

        ecg_filtered = band_pass_filter(ecg_cleaned, lowcut, highcut, fs)
        _, emrich2023 = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling, method="emrich2023")

        adjusted_rpeaks = {key: value + start_row for key, value in emrich2023.items() if key == "ECG_R_Peaks"}
        all_rpeaks.append(adjusted_rpeaks)

    combined_rpeaks = np.concatenate([r["ECG_R_Peaks"] for r in all_rpeaks if "ECG_R_Peaks" in r])
    print(f"Combined R-peaks for {file_path}: {combined_rpeaks}")
    n=len(combined_rpeaks)
    total=n/total_time
    avg_hr=total*60
    print(avg_hr)
    if combined_rpeaks.size > 0:
        hrv_results = nk.hrv(combined_rpeaks, sampling_rate=sampling, show=False)
        hrv_results['File_Name'] = file_name
        return hrv_results

    return None

file_ranges = {
    "103_ECG.csv": (0, 129000),
    "105_ECG.csv": (0, 88000),
    "106_ECG.csv": (45000,153620),
    "115_ECG.csv": (94000, 170813),
    "118_ECG.csv": (0, 140000),
    "120_ECG.csv": (3000, 177346),
    "141_ECG.csv": (78000, 162000),
    "143_ECG.csv": (0, 80000),
    "144_ECG.csv": (0, 105000),
    "147_ECG.csv": (0, 195000),
    "148_ECG.csv": (0, 105000),
    "157_ECG.csv": (0, 150000),
    "160_ECG.csv": (52000, 111827),
    "161_ECG.csv": (0, 165000),
    "164_ECG.csv": (0, 75000),
    "169_ECG.csv": (56000, 120000),
    "171_ECG.csv": (12000, 131583),
    "181_ECG.csv": (0, 64000),
    "184_ECG.csv": (0, 225000),
    "194_ECG.csv": (0, 105000),
    "204_ECG.csv": (0, 74000),
    "209_ECG.csv": (0, 75000),
    "214_ECG.csv": (45000, 190736)
}

folder_path = '/content/drive/MyDrive/new_folder'
all_hrv_results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        if filename in file_ranges:
            start, end = file_ranges[filename]
            print(f"Processing file: {file_path} with range {start} to {end}")
            hrv_result = process_ecg_file(file_path, start, end)
        else:
            print(f"Processing file: {file_path} with full range")
            hrv_result = process_ecg_file(file_path)

        if hrv_result is not None:
            all_hrv_results.append(hrv_result)

if all_hrv_results:
    combined_hrv_data = pd.concat(all_hrv_results, ignore_index=True)
    combined_hrv_data.to_csv('combined_hrv_data.csv', index=False)

    # Create box plots for each HRV feature
    hrv_features = combined_hrv_data.columns.difference(['File_Name'])
    for feature in hrv_features:
        plt.figure(figsize=(10, 6))
        combined_hrv_data.boxplot(column=feature)
        plt.title(f'Box Plot for {feature}')
        plt.ylabel('Values')
        plt.show()

# Create box plots for each HRV feature on a single page
    hrv_features = combined_hrv_data.columns.difference(['File_Name'])
    num_features = len(hrv_features)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(hrv_features):
        combined_hrv_data.boxplot(column=feature, ax=axes[i])
        axes[i].set_title(f'Box Plot for {feature}')
        axes[i].set_ylabel('Values')

    for i in range(num_features, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

"""### **hrv using cleaned data**"""

import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
import numpy as np

def process_ecg_file(file_path, start=None, end=None, chunk_size=15000):
    read_file = pd.read_csv(file_path)
    if 'time' not in read_file.columns or read_file['time'].dropna().empty:
        print(f"Skipping file {file_path} as 'time' column has no data.")
        return None

    if start is not None and end is not None:
        read_file = read_file.iloc[start:end]

    read_file['time'] = pd.to_datetime(read_file['time'])
    total_time = (read_file['time'].max() - read_file['time'].min()).total_seconds()
    print(f"Total time for {file_path}: {total_time:.2f} seconds")

    n = len(read_file) // total_time
    sampling = int(n)
    print(f"Sampling rate for {file_path}: {sampling}")

    num_chunks = len(read_file) // chunk_size
    all_rpeaks = []

    file_name = os.path.basename(file_path)
    if '101_ECG' <= file_name <= '175_ECG':
        column_index = 2
    else:
        column_index = 1

    for i in range(num_chunks + 1):
        start_row = i * chunk_size
        end_row = (i + 1) * chunk_size
        ecg_signal = read_file.iloc[start_row:end_row, column_index].values

        if len(ecg_signal) == 0:
            continue

        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling, method='vg')

        lowcut = 4.0
        highcut = 50.0
        fs = sampling

        def band_pass_filter(ecg_cleaned, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(order, [low, high], btype='band')
            padlen = min(len(ecg_cleaned) - 1, int(0.25 * len(ecg_cleaned)))
            y = signal.filtfilt(b, a, ecg_cleaned, padlen=padlen)
            return y

        ecg_filtered = band_pass_filter(ecg_cleaned, lowcut, highcut, fs)
        _, emrich2023 = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling, method="emrich2023")

        adjusted_rpeaks = {key: value + start_row for key, value in emrich2023.items() if key == "ECG_R_Peaks"}
        all_rpeaks.append(adjusted_rpeaks)

    combined_rpeaks = np.concatenate([r["ECG_R_Peaks"] for r in all_rpeaks if "ECG_R_Peaks" in r])
    print(f"Combined R-peaks for {file_path}: {combined_rpeaks}")
    if combined_rpeaks.size > 0:
        total_rpeaks = len(combined_rpeaks)
        total_1min= total_rpeaks/total_time
        avg_hr = total_1min*60
        print(avg_hr)

        hrv_results = nk.hrv(combined_rpeaks, sampling_rate=sampling, show=False)
        hrv_results['File_Name'] = file_name
        hrv_results['HR'] = avg_hr

        csv_file = 'combined_hrv.csv'
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file)
            updated_data = pd.concat([existing_data, hrv_results], ignore_index=True)
            updated_data.to_csv(csv_file, index=False)
        else:
            hrv_results.to_csv(csv_file, index=False)
        print(hrv_results)

    return combined_rpeaks

file_ranges = {
    "103_ECG.csv": (0, 129000),
    "105_ECG.csv": (0, 88000),
    "106_ECG.csv": (45000,153620),
    "115_ECG.csv": (94000, 170813),
    "118_ECG.csv": (0, 140000),
    "120_ECG.csv": (3000, 177346),
    "141_ECG.csv": (78000, 162000),
    "143_ECG.csv": (0, 80000),
    "144_ECG.csv": (0, 105000),
    "147_ECG.csv": (0, 195000),
    "148_ECG.csv": (0, 105000),
    "157_ECG.csv": (0, 150000),
    "160_ECG.csv": (52000, 111827),
    "161_ECG.csv": (0, 165000),
    "164_ECG.csv": (0, 75000),
    "169_ECG.csv": (56000, 120000),
    "171_ECG.csv": (12000, 131583),
    "181_ECG.csv": (0, 64000),
    "184_ECG.csv": (0, 225000),
    "194_ECG.csv": (0, 105000),
    "204_ECG.csv": (0, 74000),
    "209_ECG.csv": (0, 75000),
    "214_ECG.csv": (45000, 190736)
}

folder_path = '/content/drive/MyDrive/new_folder'
all_hrv_results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        if filename in file_ranges:
            start, end = file_ranges[filename]
            print(f"Processing file: {file_path} with range {start} to {end}")
            hrv_result = process_ecg_file(file_path, start, end)
        else:
            print(f"Processing file: {file_path} with full range")
            hrv_result = process_ecg_file(file_path)





