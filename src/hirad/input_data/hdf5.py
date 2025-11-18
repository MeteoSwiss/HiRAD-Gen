from h5py import File
import os
import torch
import logging
import numpy as np

IN_DIR = '/store_new/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-all-channels/era-interpolated'
DB_FILENAME='/store_new/mch/msopr/hirad-gen/input-data-hdf5.db'

class HiradHdf5:
	def __init__(self, in_dir: str, db_path: str, data_shape: tuple, to_write = False):
		self.in_dir = in_dir
		self.files = sorted(os.listdir(in_dir))
		self.db_path = db_path
		shape = list(data_shape)
		shape.insert(0,len(self.files))
		self.db_shape = tuple(shape)
		self.to_write = to_write
		if to_write:
			self.dbf = File(DB_FILENAME, "w")
			self.dset = self.dbf.create_dataset("all-channels", self.db_shape, dtype='float64')
		else:
			self.dbf = File(DB_FILENAME, "r")
			self.dset = self.dbf.require_dataset("all-channels", self.db_shape, dtype='float64', exact=True)
		return
	
	def fill_database(self):
		if not self.to_write:
			raise PermissionError('database not opened with write')
		logging.info('filling database')
		for i in range(len(self.files)):
		#for i in range(10):
			f = self.files[i]
			logging.info(f'saving {f}')
			torchdata = torch.load(os.path.join(IN_DIR, f), weights_only=False)
			self.dset[i,:] = torchdata

	def read_index(self, i: int):
		return self.dset[i,:]

	def read_datetime(self, datefmt: str):
		# Raises ValueError if not found
		i = self.files.index(datefmt)
		return self.read_index(i)

	def read_database(self):
		for i in range(10):
			nparray = self.dset[i,:]
			f = self.files[i]
			torchdata = torch.load(os.path.join(IN_DIR, f), weights_only=False)
			logging.info(np.array_equal(torchdata, nparray))
	
def main():
	logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 
	# write DB
	hdf5db = HiradHdf5(IN_DIR, DB_FILENAME, (101, 1, 191488), True)
	hdf5db.fill_database()
	# read DB
	hdf5db = HiradHdf5(IN_DIR, DB_FILENAME, (101, 1, 191488), False)
	hdf5db.read_database()
	hdf5db.read_index(0)
	hdf5db.read_datetime('20160101-0900')
	#read_database()
	#nparray = get_data_for_time('20160101-0900')
	#print(nparray.dtype)


if __name__ == "__main__":
	main()