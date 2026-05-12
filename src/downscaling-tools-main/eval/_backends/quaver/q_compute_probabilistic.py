import os
import pandas as pd
import argparse

# DATETIME = int(os.environ["YMD"]) * 100 + int(os.environ["TIME"]) // 100
# DATETIME_o = int(DATETIME/100)

#########################
# q_compute_probabilistic
#########################


parser = argparse.ArgumentParser(description="Process forecast verification parameters")

parser.add_argument("--expver", type=str, default="ioac", help="Experiment version")

parser.add_argument("--nmem", type=int, default=2, help="Number of members")
parser.add_argument(
    "--first_reference_date",
    type=int,
    default=20250201,
    help="First reference date (YYYYMMDDHH)",
)
parser.add_argument(
    "--last_reference_date",
    type=int,
    default=20250225,
    help="Last reference date (YYYYMMDDHH)",
)
parser.add_argument(
    "--date_step", type=int, default=24, help="Step between dates in hours"
)
parser.add_argument(
    "--first_lead_time", type=int, default=24, help="First lead time in hours"
)
parser.add_argument(
    "--last_lead_time", type=int, default=240, help="Last lead time in hours"
)
parser.add_argument(
    "--lead_time_step", type=int, default=24, help="Step between lead times in hours"
)
parser.add_argument("--grid", type=str, default="O320", help="Grid specification")
parser.add_argument(
    "--class", dest="class_", type=str, default="rd", help="Class identifier"
)
parser.add_argument("--database", type=str, default="fdb", help="Database name")


args = parser.parse_args()

# Instantiate all parameters with their original variable names
nmem = args.nmem
first_reference_date = args.first_reference_date
last_reference_date = args.last_reference_date
date_step = args.date_step
first_lead_time = args.first_lead_time
last_lead_time = args.last_lead_time
lead_time_step = args.lead_time_step
grid = args.grid
class_ = args.class_
expver = args.expver
database = args.database
oro_interp_postproc = (
    "orography_correction:"
    f"class=od,number=1,stream=enfo,type=pf,step=0,expver=0001,date=2024-01-01,grid={grid},database=off"
)


date1 = first_reference_date
date2 = last_reference_date

nmem_verif = nmem

sfc_param = ["2t", "10ff", "2d"]
pl_param = ["t", "z", "u", "v"]
levelist = [1000, 850, 500]


def get_forecast_quaver_ens(DATETIME, numbers):
    return forecast(
        date=DATETIME,
        step=StepSequence(first_lead_time, last_lead_time, lead_time_step),
        Class=f"{class_ }",
        expver=f"{expver }",
        database=f"{database }",
        number=numbers,
        stream="enfo",
        type="pf",
    )


def analysis_upperair(DATETIME, preproc, numbers):
    forecast_quaver = get_forecast_quaver_ens(DATETIME, numbers)
    compute(
        forecast=forecast_quaver,
        reference=analysis(Class="od", expver="0001"),
        specifics=specifics(
            score=["rmsef", "ccaf", "sdaf"],
            domain=["n.hem", "tropics", "s.hem", "europe"],
            grid=[1.5, 1.5],
            truncation="120",
            levtype=["pl"],
            parameter=pl_param,
            levelist=levelist,
        ),
        preprocess=preproc,
        vstream=f"prepml_{expver}_an",
        overwrite="yes",
        ignore_missing="no",
    )

    if preproc:
        compute(
            forecast=forecast_quaver,
            reference=analysis(Class="od", expver="0001"),
            specifics=specifics(
                score=["spread", "fcrps", "crps"],
                domain=["n.hem", "tropics", "s.hem", "europe"],
                grid=[1.5, 1.5],
                truncation="120",
                levtype=["pl"],
                parameter=pl_param,
                levelist=levelist,
            ),
            vstream=f"prepml_{expver}_an",
            overwrite="yes",
            ignore_missing="no",
        )


def observations_surface(DATETIME, preproc, numbers):
    forecast_quaver = get_forecast_quaver_ens(DATETIME, numbers)
    forecast_quaver_accumulation = forecast(
        date=DATETIME,
        step=StepSequence(first_lead_time, last_lead_time, lead_time_step),
        Class=f"{class_ }",
        expver=f"{expver}",
        database=f"{ database}",
        number=numbers,
        stream="enfo",
        type="pf",
    )

    all_parameters = sfc_param
    parameters_without_precip = [p for p in all_parameters if p not in ["tp"]]

    all_scores_det = ["rmsef", "seeps"]
    all_scores_det_without_seeps = [p for p in all_scores_det if p not in ("seeps",)]

    all_scores_ens = ["fcrps", "crps", "spread"]

    common_specifics = dict(
        levtype="sfc",
        grid=grid,
        truncation="none",
        domain=["n.hem", "tropics", "s.hem", "europe"],
    )
    common_mem = dict(
        preprocess=preproc,
        vstream=f"prepml_{expver}_ob",
        spatial_mean_weights="station_density",
        ignore_missing="no",
        overwrite="yes",
    )
    if len(parameters_without_precip) > 0:
        compute(
            reference=surfaceobservations(),
            interpolation_postprocessor=oro_interp_postproc,
            forecast=forecast_quaver,
            specifics=specifics(
                parameter=parameters_without_precip,
                score=all_scores_det_without_seeps,
                vstream=f"prepml_{expver}_ob",
                **common_specifics,
            ),
            **common_mem,
        )
    if "tp" in all_parameters:
        if len(all_scores_det_without_seeps) > 0:
            compute(
                reference=surfaceobservations(climatology=stationclimatology()),
                forecast=forecast_quaver_accumulation,
                specifics=specifics(
                    parameter="tp",
                    period=24,
                    score=all_scores_det_without_seeps,
                    vstream=f"prepml_{expver}_ob",
                    **common_specifics,
                ),
                **common_mem,
            )
        compute(
            reference=surfaceobservations(
                observation_filter=["toss", "seeps_filter"],
                climatology=stationclimatology(),
            ),
            forecast=forecast_quaver_accumulation,
            specifics=specifics(
                parameter="tp",
                period=24,
                score=["seeps"],
                vstream=f"prepml_{expver}_ob",
                **common_specifics,
            ),
            **common_mem,
        )

    if preproc:
        common_ens = dict(
            vstream=f"prepml_{expver}_ob",
            spatial_mean_weights="station_density",
            ignore_missing="no",
            overwrite="yes",
        )

        if len(parameters_without_precip) > 0:
            compute(
                reference=surfaceobservations(climatology=stationclimatology()),
                interpolation_postprocessor=oro_interp_postproc,
                forecast=forecast_quaver,
                specifics=specifics(
                    parameter=parameters_without_precip,
                    score=all_scores_ens,
                    vstream=f"prepml_{expver}_ob",
                    **common_specifics,
                ),
                **common_ens,
            )

        if "tp" in all_parameters:
            compute(
                reference=surfaceobservations(climatology=stationclimatology()),
                forecast=forecast_quaver_accumulation,
                specifics=specifics(
                    parameter="tp",
                    period=24,
                    score=all_scores_ens,
                    vstream=f"prepml_{expver}_ob",
                    **common_specifics,
                ),
                **common_ens,
            )


run = 0
dates = pd.date_range("%s %s:00" % (date1, run), "%s %s:00" % (date2, run), freq="1D")

ensemble_number_list = list(range(1, nmem + 1))
for date in dates:
    DATETIME = (date.year * 10000 + date.month * 100 + date.day) * 100 + date.hour

    observations_surface(
        DATETIME,
        preproc=[
            "mean",
        ],
        numbers=ensemble_number_list,
    )

    for num in ensemble_number_list:
        observations_surface(DATETIME, preproc=[], numbers=int(num))

    analysis_upperair(
        DATETIME,
        preproc=[
            "mean",
        ],
        numbers=ensemble_number_list,
    )
    for num in ensemble_number_list:
        analysis_upperair(DATETIME, preproc=[], numbers=int(num))
