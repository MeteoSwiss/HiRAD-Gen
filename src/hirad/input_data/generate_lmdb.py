import lmdb
import os
import torch

print('starting')
in_dir = '/store_new/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-all-channels/era-interpolated'
filename='/store_new/mch/msopr/hirad-gen/all-channels-lmdb.db'

lmdb_env = lmdb.open(filename, map_size=int(1e10))
lmdb_txn = lmdb_env.begin(write=True)
files = sorted(os.listdir(in_dir))
for i in range(100):
	f = files[i]
	print(i)
	torchdata = torch.load(os.path.join(in_dir, f), weights_only=False)
	lmdb_txn.put(f.encode(), torchdata)
lmdb_txn.commit()