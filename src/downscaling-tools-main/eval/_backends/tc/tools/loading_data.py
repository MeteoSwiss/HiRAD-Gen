import os
import numpy as np
import metview as mv
from icecream import ic
import earthkit.data as ekd


class DataRetriever:
    def __init__(
        self, main_dir, dates, year, month, resol, lat_down, lat_up, long_down, long_up
    ):
        self.main_dir = main_dir
        self.dates = dates
        self.year = year
        self.month = month
        self.resol = resol
        self.lat_down = lat_down
        self.lat_up = lat_up
        self.long_down = long_down
        self.long_up = long_up

    def read_surface_series(self, expid, prefix):
        n = len(self.dates)
        holder = np.empty(n, dtype=object)
        for i, day in enumerate(self.dates):
            date = f"{self.year}{self.month}{day:02}"
            ic(f"{self.main_dir}/{prefix}_{expid}_{date}.grib")
            holder[i] = mv.read(f"{self.main_dir}/{prefix}_{expid}_{date}.grib")
        return holder

    def retrieve_data(self, holder, parameter, is_pf=True):
        def process_data(data, param):
            return mv.read(
                data=data,
                grid=[self.resol, self.resol],
                area=[self.lat_down, self.long_down, self.lat_up, self.long_up],
                param=param,
            )

        def adjust_units(result, param):
            if param == "msl":
                return result / 100
            elif param in ["2t", "2d"]:
                return result - 273.13
            elif param == "tp":
                return result * 1000
            return result

        output = None
        for i, data in enumerate(holder):
            if parameter == "10m wind":
                u10 = process_data(data, "10u").to_dataset()["u10"].values
                v10 = process_data(data, "10v").to_dataset()["v10"].values
                variable = np.sqrt(u10**2 + v10**2)
            else:
                result = process_data(data, parameter)
                result = adjust_units(result, parameter)
                param_key = {"2t": "t2m", "2d": "d2m"}.get(parameter, parameter)
                variable = result.to_dataset()[param_key].values

            shape = variable.shape
            if is_pf:
                if output is None:
                    output = np.empty((len(holder), *shape))
                output[i] = variable[:, :, ::-1, :] if len(shape) == 4 else variable
                ic(
                    f"pf - Output shape [days, members, lead_times, resol1-2]: {output.shape}"
                )

            else:
                if output is None:
                    output = np.empty((len(holder), *shape))
                output[i] = variable[:, ::-1, :]
                ic(f"cf - Output shape [days, times, resol1-2]: {output.shape}")

        return output

    def retrieve_all_data(
        self,
        analysis,
        expid_enfo_O320,
        expid_eefo_O96,
        list_expid_ml,
    ):
        # Read data holders
        analysis_O320_holder = self.read_surface_series(analysis, "surface_an")
        enfo_O320_holder = self.read_surface_series(expid_enfo_O320, "surface_pf")
        eefo_O96_holder = self.read_surface_series(expid_eefo_O96, "surface_pf")
        ml_holders = {
            expid: self.read_surface_series(expid, "surface_pf")
            for expid in list_expid_ml
        }

        # Retrieve MSL
        ic("Retrieving mean sea level pressure (MSL) data...")
        msl = {
            "OPER_O320_0001": self.retrieve_data(
                analysis_O320_holder, "msl", is_pf=False
            ),
            "ENFO_O320_0001": self.retrieve_data(enfo_O320_holder, "msl"),
            "EEFO_O96_0001": self.retrieve_data(eefo_O96_holder, "msl"),
            **{
                expid: self.retrieve_data(holder, "msl")
                for expid, holder in ml_holders.items()
            },
        }

        # Retrieve wind speed
        ic("Retrieving 10m wind speed data...")
        wind10m = {
            "OPER_O320_0001": self.retrieve_data(
                analysis_O320_holder, "10m wind", is_pf=False
            ),
            "ENFO_O320_0001": self.retrieve_data(enfo_O320_holder, "10m wind"),
            "EEFO_O96_0001": self.retrieve_data(eefo_O96_holder, "10m wind"),
            **{
                expid: self.retrieve_data(holder, "10m wind")
                for expid, holder in ml_holders.items()
            },
        }

        # Convert analysis output starting at +24h to match the model output
        wind10m["OPER_O320_0001"] = wind10m["OPER_O320_0001"][:, 1:, :, :]
        msl["OPER_O320_0001"] = msl["OPER_O320_0001"][:, 1:, :, :]

        return msl, wind10m
