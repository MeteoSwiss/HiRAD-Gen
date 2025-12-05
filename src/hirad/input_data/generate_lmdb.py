import lmdb
import os
import torch
import numpy as np

IN_DIR = '/store_new/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-all-channels/era-interpolated'
DB_FILENAME='/store_new/mch/msopr/hirad-gen/all-channels-lmdb.db'

class HiradLmdb:
	def __init__(self, db_path: str, to_write = False):
		self.env = lmdb.open(db_path)
		self.txn = self.env.begin(write=to_write)
		self.cursor = self.txn.cursor()
		return
	
	def get_next(self) -> (str, np.ndarray):
		return key, nparray
	

def fill_database():
	print('starting')

	lmdb_env = lmdb.open(DB_FILENAME, map_size=int(1e11))
	lmdb_txn = lmdb_env.begin(write=True)
	files = sorted(os.listdir(IN_DIR))
	#for i in range(len(files)):
	for i in range(10):
		f = files[i]
		print(f)
		torchdata = torch.load(os.path.join(IN_DIR, f), weights_only=False)
		lmdb_txn.put(f.encode(), torchdata)
		print(torchdata.shape)
		print(torchdata.dtype)
	lmdb_txn.commit()
	lmdb_env.close()

def get_data_for_time(datefmt: str):
	lmdb_env = lmdb.open(DB_FILENAME)
	lmdb_txn = lmdb_env.begin(write=False)
	data = lmdb_txn.get(datefmt.encode())
	if data == None:
		raise KeyError(f'date {datefmt} not found in database')
	data = np.frombuffer(data, dtype=np.float64).reshape([101,1,191488])
	lmdb_txn.commit()
	lmdb_env.close()
	return data



def read_database():
	lmdb_env = lmdb.open(DB_FILENAME)
	lmdb_txn = lmdb_env.begin(write=False)
	lmdb_cursor = lmdb_txn.cursor()
	nsamples = 0
	for key, value in lmdb_cursor:
		print (type(key))
		print(key)
		print (type(value))
		nsamples = nsamples + 1
	print(nsamples)
	
	

def main():
	fill_database()
	read_database()
	nparray = get_data_for_time('20160101-0900')
	print(nparray.dtype)


if __name__ == "__main__":
	main()