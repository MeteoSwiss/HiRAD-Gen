import numpy as np
import yaml
import os
from tqdm import tqdm
import torch


def transform_channel(channel_array, channel_name="tp"):
    if channel_name == "tp":
        channel_array = np.clip(channel_array, 0, None)
        channel_array = (np.power(channel_array,0.25)-1)/0.25
    return channel_array

def main():
    base_path = '/iopsstor/scratch/cscs/mmcgloho/basic-numpy/era5-realch1/v1.0-channel-subset/train'
    all_files_input = os.listdir(os.path.join(base_path, 'era-copernicus-interpolated/'))
    all_files_output = os.listdir(os.path.join(base_path, 'realch1/'))
    input_info = yaml.safe_load(open(os.path.join(base_path, 'info', 'era.yaml')))
    output_info = yaml.safe_load(open(os.path.join(base_path, 'info', 'realch1.yaml')))

    tp_input_index = input_info['select'].index('tp')
    tp_output_index = output_info['select'].index('tp')

    print('Total precipitation input index:', tp_input_index)
    print('Total precipitation output index:', tp_output_index)

    print(f"Found {len(all_files_input)} input files and {len(all_files_output)} output files.")
    print("Calculating transformed mean and std...")

    # calculate mean
    input_values = []
    output_values = []
    print("Calculating input mean")
    for f in tqdm(all_files_input):
        data = np.load(os.path.join(base_path, 'era-copernicus-interpolated', f))
        data = transform_channel(data[tp_input_index,:])
        input_values.append(np.mean(data))
    input_mean = np.mean(input_values)
    print("Calculating output mean")
    for f in tqdm(all_files_output):
        data = np.load(os.path.join(base_path, 'realch1', f))
        data = transform_channel(data[tp_output_index,:])
        output_values.append(np.mean(data))
    output_mean = np.mean(output_values)
    
    torch.save(input_mean, os.path.join('./info', 'era5-tp-box_cox_025-mean'))
    torch.save(output_mean, os.path.join('./info','realch1-tp-box_cox_025-mean'))

    # calculate std
    input_values = []
    output_values = []
    print("Calculating input std")
    for f in tqdm(all_files_input):
        data = np.load(os.path.join(base_path, 'era-copernicus-interpolated', f))
        data = transform_channel(data[tp_input_index,:])
        input_values.append(np.mean((data - input_mean)**2))
    input_std = np.sqrt(np.mean(input_values))
    print("Calculating output std")
    for f in tqdm(all_files_output):
        data = np.load(os.path.join(base_path, 'realch1', f))
        data = transform_channel(data[tp_output_index,:])
        output_values.append(np.mean((data - output_mean)**2))
    output_std = np.sqrt(np.mean(output_values))

    torch.save(input_std, os.path.join('./info','era5-tp-box_cox_025-std'))
    torch.save(output_std, os.path.join('./info','realch1-tp-box_cox_025-std'))

    print(f"Input Mean: {input_mean}, Input Std: {input_std}")
    print(f"Output Mean: {output_mean}, Output Std: {output_std}")
    print("Done.")

if __name__ == "__main__":
    main()