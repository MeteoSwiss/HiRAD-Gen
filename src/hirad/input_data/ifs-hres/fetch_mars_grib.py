import argparse
from datetime import datetime, timedelta

from ecmwfapi import ECMWFService


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch IFS HRES GRIB data from MARS over a date range.")
    parser.add_argument("--start-date", required=True, type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                         help="First date to fetch, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                         help="Last date to fetch (inclusive), in YYYY-MM-DD format.")
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date must not be before --start-date")
    return args


args = parse_args()
START_DATE = args.start_date
END_DATE = args.end_date

server = ECMWFService("mars")


# --- CONFIGURATION ---
CHUNK_SIZE_SFC = 5 # 5 days keeps the 450GB month safely under the 75GB per-request limit
CHUNK_SIZE_PL = 1  # PL data has only 2 base times per tape, so 1 day avoids the queuing penalty for >1 tape requests.

TARGET_DIR = '/capstor/scratch/cscs/mmcgloho/mars/'

# Common settings shared between both runs
base_request = {
    "class": "od",
    "type": "fc",
    "stream": "oper",
    "expver": "1",
    "time": "00:00:00/12:00:00",
    "grid": "o1280",
    # ALL AVAILABLE STEPS SYNTAX (Out to 15 days / 360 hours)
    "step": (
            "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/"
            "25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/"
            "49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/"
            "73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/" # Hourly to 90
            "93/96/99/102/105/108/111/114/117/120/123/126/129/132/135/138/141/144/" # 3-Hourly to 144
            "150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/"
            "258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360" # 6-Hourly to 360
            ),
        
}

# Surface level data

sfc_chunk_start = START_DATE
while sfc_chunk_start <= END_DATE:
    sfc_chunk_end = min(sfc_chunk_start + timedelta(days=CHUNK_SIZE_SFC - 1), END_DATE)

    sfc_chunk_start_date = sfc_chunk_start.isoformat()
    sfc_chunk_end_date = sfc_chunk_end.isoformat()

    print(f"processing chunk {sfc_chunk_start_date} to {sfc_chunk_end_date}")

    # --- REQUEST 1: SURFACE ---
    print("Submitting surface level request...")
    sfc_request = base_request.copy()
    sfc_request.update({
        "date": f"{sfc_chunk_start_date}/to/{sfc_chunk_end_date}",
        "levtype": "sfc",
        "param": "165.128/166.128/167.128/228.128/136.128", # Wind, Temp, Total Precip, Total Column Water
    })
    server.execute(sfc_request, TARGET_DIR + f"ifs_operational_surface_{sfc_chunk_start_date}-{sfc_chunk_end_date}.grib")

    sfc_chunk_start = sfc_chunk_end + timedelta(days=1)

pl_chunk_start = START_DATE
while pl_chunk_start <= END_DATE:
    # --- REQUEST 2: PRESSURE LEVELS FOR SHORT RANGE ---
    pl_chunk_end = min(pl_chunk_start + timedelta(days=CHUNK_SIZE_PL - 1), END_DATE)

    pl_chunk_start_date = pl_chunk_start.isoformat()
    pl_chunk_end_date = pl_chunk_end.isoformat()

    print(f"processing chunk {pl_chunk_start_date} to {pl_chunk_end_date}")

    print("Submitting pressure levels request for short lead times...")
    pl_request_short = base_request.copy()
    pl_request_short.update({
        "date": f"{pl_chunk_start_date}/to/{pl_chunk_end_date}",
        "levtype": "pl",
        "grid": "av", # archive value-- avoids heavy post-processing on MARS side
        "step": (
            "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/"
            "25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/"
            "49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/"
            "73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/" # Hourly to 90
            "93/96/99/102/105/108/111/114/117/120/123/126/129/132/135/138/141/144" # 3-Hourly to 144
        ),
        "levelist": "850/500",         # Specify vertical layers
        "param": "129/130/138/155",  # Geopotential, Temp, Vorticity, Divergence
        #"param": "129/130/131/132",    # Geopotential, Temp, U/V Wind
    })
    server.execute(pl_request_short, TARGET_DIR + f"ifs_operational_pl_av_short_{pl_chunk_start_date}-{pl_chunk_end_date}.grib")


    # --- REQUEST 3: PRESSURE LEVELS FOR LONG RANGE ---
    pl_request_long = base_request.copy()
    pl_request_long.update({
        "date": f"{pl_chunk_start_date}/to/{pl_chunk_end_date}",
        "levtype": "pl",
        "grid": "av", # archive value-- avoids heavy post-processing on MARS side
        "step": (
            "150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/"
            "246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/"
            "348/354/360" # 6-hourly to 360
         ),
        "levelist": "850/500",         # Specify vertical layers
        "param": "129/130/138/155",  # Geopotential, Temp, Vorticity, Divergence
        #"param": "129/130/131/132",    # Geopotential, Temp, U/V Wind
    })
    server.execute(pl_request_long, TARGET_DIR + f"ifs_operational_pl_av_long_{pl_chunk_start_date}-{pl_chunk_end_date}.grib")

    pl_chunk_start = pl_chunk_end + timedelta(days=1)


print("All downloads complete!")

