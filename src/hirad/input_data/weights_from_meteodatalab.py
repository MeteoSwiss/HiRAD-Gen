import xarray as xr
import numpy as np

ds_1km = xr.open_dataset('/capstor/store/mch/msopr/hirad-gen/meteodatalab-output/icon-ch1-eps-rotlatlon.nc')
np.save('/capstor/store/cscs/pasc/c38/real2cosmo_grid_info/remap_weights_1km.npy', ds_1km['rbf_B_wgt'].values)
np.save('/capstor/store/cscs/pasc/c38/real2cosmo_grid_info/remap_indices_1km.npy', ds_1km['rbf_B_glbidx'].values)
ds_1km.close()

ds_2km = xr.open_dataset('/capstor/store/mch/msopr/hirad-gen/meteodatalab-output/icon-ch1-eps-to-ch2grid-rotlatlon.nc')
new_weights = ds_2km['rbf_B_wgt'].values
new_indices = ds_2km['rbf_B_glbidx'].values

existing_weights = np.load('/capstor/store/cscs/pasc/c38/real2cosmo_grid_info/remap_weights_2km.npy')
existing_indices = np.load('/capstor/store/cscs/pasc/c38/real2cosmo_grid_info/remap_indices_2km.npy')
print('weights match existing:', np.allclose(new_weights, existing_weights))
print('indices match existing:', np.array_equal(new_indices, existing_indices))

np.save('/capstor/store/cscs/pasc/c38/real2cosmo_grid_info/remap_weights_2km.npy', new_weights)
np.save('/capstor/store/cscs/pasc/c38/real2cosmo_grid_info/remap_indices_2km.npy', new_indices)
ds_2km.close()