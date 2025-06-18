import yaml
from anemoi.datasets import open_dataset
from anemoi.datasets import save_dataset
import shutil
import pathlib

#with open('src/input_data/era-1h.yaml') as era_file:
#    era_config=yaml.safe_load(era_file)

INPUT = 'aifs-ea-an-oper-0001-mars-n320-1979-2023-1h-v1-with-ERA51'
OUTPUT = '/store_new/mch/msopr/hirad-gen/era5-1h-subset.zarr'

#era=open_dataset(INPUT)
#era=open_dataset(era,start='2015-11-29', end='2020-10-28',area=(50.98, -0.94613, 41.600636, 17.846238))
# Delete failed initialization, if exists
if pathlib.Path.exists(pathlib.Path(OUTPUT)):
    shutil.rmtree(OUTPUT)
save_dataset({"dataset":INPUT,
              "start":"2016-01-01",
              "end":"2016-01-01",
              "select":['2t', '10u', '10v', 'tcw', 't_850', 'z_850', 'u_850', 'v_850', 't_500', 'z_500', 'u_500', 'v_500','tp'],
              }, OUTPUT,5)


#save_dataset(era, OUTPUT,
