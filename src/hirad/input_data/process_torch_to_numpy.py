
import os
import sys

import numpy as np
import torch


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    in_files = os.listdir(input_dir)
    in_files.sort()
    for i in range(10):
        f = in_files[i]
        print(f)
    #for f in in_files:
        data = torch.load(os.path.join(input_dir, f), weights_only=False)
        np.save(os.path.join(output_dir, f), data)

if __name__ == "__main__":
    main()

