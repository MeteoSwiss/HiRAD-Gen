from hirad.utils.inference_utils import save_as_grib
import torch

target=torch.load('/capstor/scratch/cscs/pstamenk/outputs/generation/generation_era_cosmo_anemoi_JJA_2020/20200601-0000/20200601-0000-target',weights_only=False)
save_as_grib('myfile.grib','/users/mmcgloho/evalml/resources/inference/templates', ['2t', '10u', '10v', 'tp'], target, 'co2')