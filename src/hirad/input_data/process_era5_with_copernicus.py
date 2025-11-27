import logging
import os
import sys

import numpy as np
import torch
import yaml

import interpolate_basic

def load(filename: str):
    if filename.endswith('.npy'):
        return np.load(filename)
    return torch.load(filename, weights_only=False)

def save(values: np.ndarray, filename: str):
    if filename.endswith('.npy'):
        np.save(filename, values)
    else:
        torch.save(values, filename)
    return

def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call process_copernicus_era.py [era.yaml] [copernicus.yaml] [path]')
    infile_era = sys.argv[1]
    infile_copernicus = sys.argv[2]
    path = sys.argv[3]

    os.makedirs(os.path.join(path, 'era-copernicus-interpolated'), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(path, f'process-era-copernicus.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 

    logging.info(f'running {sys.argv}')

    # Extract ERA channels from input yaml
    with open(infile_era) as era_cfg_file:
        era_config = yaml.safe_load(era_cfg_file)
    with open(infile_copernicus) as cop_cfg_file:
        copernicus_config = yaml.safe_load(cop_cfg_file)

    era_channels = era_config['select']
    copernicus_channels = copernicus_config['channels']
    mapping = {}
    for c in range(len(copernicus_channels)):
        if era_channels.count(copernicus_channels[c]) == 1:            
            e = era_channels.index(copernicus_channels[c])
            mapping[e] = c
    logging.info('replacing {len(mapping)} channels in ERA data: {mapping}')

    era_dir = os.path.join(path, 'era-interpolated')
    copernicus_dir = os.path.join(path, 'copernicus-interpolated')
    output_dir = os.path.join(path, 'era-copernicus-interpolated')
    era_files = os.listdir(era_dir)
    copernicus_files = os.listdir(copernicus_dir)
    intersect_files = list(set(era_files).intersection(set(copernicus_files)))

    for f in intersect_files:
        era = load(os.path.join(era_dir, f))
        copernicus = load(os.path.join(copernicus_dir, f))
        assert np.array_equal(era.shape[1:], copernicus.shape[1:]), 'Era and Copernicus files appear to have different grid shapes'
        for k,v in mapping.items():
            era[k,:] = copernicus[v,:]
        save(era, os.path.join(output_dir, f))

if __name__ == "__main__":
    main()
