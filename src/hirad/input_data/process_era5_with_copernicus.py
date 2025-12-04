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
    era_dir = sys.argv[3]
    copernicus_dir = sys.argv[4]
    output_path = sys.argv[5]

    os.makedirs(os.path.join(output_path, 'era-copernicus-interpolated'), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_path, f'process-era-copernicus.log'),
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

    #era_dir = os.path.join(output_path, 'era-interpolated')
    #copernicus_dir = os.path.join(path, 'copernicus-interpolated')
    output_dir = os.path.join(output_path, 'era-copernicus-interpolated')
    era_files = os.listdir(era_dir)
    copernicus_files = os.listdir(copernicus_dir)
    era_files.sort()
    copernicus_files.sort()
    #intersect_files = list(set(era_files).intersection(set(copernicus_files)))
    era_format = 'torch'
    copernicus_format = 'torch'
    if era_files[0].endswith('npy'):
        era_format = 'numpy'
    if copernicus_files[0].endswith('npy'):
        copernicus_format = 'numpy'

    #for f in era_files:
    #for i in range(15000):
    for i in range(10000,12000):
    #for i in range(12000,15000):
    #for i in range(15000,20000):
    #for i in range(20000,25000):
    #for i in range(25000,30000):
    #for i in range(30000,35000):
    #for i in range(35000,40000):
    #for i in range(40000,len(era_files)):
        f = era_files[i]
        logging.info(f'{i} {f}')
        era = load(os.path.join(era_dir, f))
        era = era.squeeze() # get rid of extra dimension, if present
        base_filename = f
        if era_format == 'numpy':
            base_filename = f[:-4]
        c_f = base_filename
        if copernicus_format == 'numpy':
            c_f = base_filename + '.npy'
        if c_f in copernicus_files:
            copernicus = load(os.path.join(copernicus_dir, c_f))
            assert np.array_equal(era.shape[1:], copernicus.shape[1:]), f'Era and Copernicus files appear to have different grid shapes: {era.shape} vs {copernicus.shape}'
            for k,v in mapping.items():
                era[k,:] = copernicus[v,:]
            save(era, os.path.join(output_dir, base_filename + '.npy'))

if __name__ == "__main__":
    main()
