"""Centralized plotting configuration for region plotting scripts."""
from __future__ import annotations

LEGACY_LOCAL_REGION_BOXES: dict[str, list[float]] = {
    "default": [40.0, 50.0, 0.0, 10.0],
    "pyrenees_alpes": [40.0, 50.0, 0.0, 10.0],
    "rocky_mountains": [35.0, 50.0, -120.0, -100.0],
    "amazon_forest": [-15.0, 5.0, -75.0, -45.0],
    "amazon_forest_core": [-25.0, 15.0, -90.0, -30.0],
    "southeast_asia": [-10.0, 20.0, 95.0, 150.0],
    "maritime_continent": [-18.0, 30.0, 85.0, 160.0],
    "west_sahara": [15.0, 30.0, -20.0, 0.0],
    "himalayas": [25.0, 40.0, 75.0, 100.0],
    "greatbarrier_reef": [-25.0, -10.0, 140.0, 155.0],
    "eastern_us": [25.0, 45.0, -90.0, -70.0],
    "eastern_us_coast": [10.0, 55.0, -110.0, -45.0],
    "humberto_atlantic": [5.0, 45.0, -80.0, -40.0],
    "caribbean_and_antilles": [8.0, 28.0, -90.0, -55.0],
    "idalia": [10.0, 40.0, -100.0, -70.0],
    "idalia_center": [18.0, 32.0, -92.0, -78.0],
    "central_africa": [-10.0, 10.0, 10.0, 30.0],
    "congo_basin": [-18.0, 18.0, 0.0, 45.0],
    "andes_central": [-50.0, -5.0, -95.0, -45.0],
    "european_arctic": [-25.0, 0.0, 75.0, 90.0],
    "rocky_mountains_central": [40.0, 45.0, -115.0, -105.0],
    "rocky_mountains_north": [45.0, 50.0, -115.0, -105.0],
    "rocky_mountains_south": [35.0, 40.0, -110.0, -100.0],
    "amazon_forest_central": [-5.0, 5.0, -75.0, -65.0],
    "amazon_forest_west": [-10.0, 0.0, -75.0, -65.0],
    "amazon_forest_east": [-10.0, 0.0, -55.0, -45.0],
    "southeast_asia_central": [0.0, 10.0, 100.0, 110.0],
    "southeast_asia_mainland": [10.0, 20.0, 100.0, 110.0],
    "southeast_asia_maritime": [-5.0, 5.0, 115.0, 125.0],
    "west_sahara_central": [20.0, 25.0, -15.0, -5.0],
    "west_sahara_coastal": [20.0, 25.0, -20.0, -10.0],
    "west_sahara_east": [20.0, 25.0, -10.0, 0.0],
    "himalayas_central": [10.0, 50.0, 55.0, 115.0],
    "himalayas_west": [30.0, 35.0, 75.0, 85.0],
    "himalayas_east": [25.0, 30.0, 90.0, 100.0],
    "greatbarrier_reef_central": [-20.0, -15.0, 145.0, 150.0],
    "greatbarrier_reef_north": [-15.0, -10.0, 145.0, 150.0],
    "greatbarrier_reef_south": [-25.0, -20.0, 150.0, 155.0],
    "eastern_us_central": [35.0, 40.0, -85.0, -75.0],
    "eastern_us_north": [40.0, 45.0, -80.0, -70.0],
    "eastern_us_south": [30.0, 35.0, -85.0, -75.0],
    "central_africa_congo": [-5.0, 5.0, 15.0, 25.0],
    "central_africa_north": [0.0, 10.0, 15.0, 25.0],
    "central_africa_south": [-10.0, 0.0, 20.0, 30.0],
}

O96_INTERESTING_REGIONS: dict[str, list[float]] = {
    # Humberto-specific: Atlantic hurricane basin and impact zone
    "humberto_atlantic": [5.0, 45.0, -80.0, -40.0],
    "eastern_us_coast": [10.0, 55.0, -110.0, -45.0],
    "caribbean_and_antilles": [8.0, 28.0, -90.0, -55.0],
    # Orographic benchmarks
    "amazon_forest_core": [-25.0, 15.0, -90.0, -30.0],
    "andes_central": [-50.0, -5.0, -95.0, -45.0],
}

O1280_INTERESTING_REGIONS: dict[str, list[float]] = {
    "tibet_karakoram": [33.5, 41.5, 79.5, 91.5],
    "andes_central": [-37.5, -29.5, -75.5, -63.5],
    "greenland_south_tip": [58.5, 66.5, -54.5, -42.5],
    "horn_of_africa": [5.5, 13.5, 35.5, 47.5],
    "iran_zagros": [31.5, 39.5, 46.5, 58.5],
    "rockies_wyoming": [40.5, 48.5, -113.5, -101.5],
    "himalayas_west": [31.5, 39.5, 64.5, 76.5],
    "new_zealand_north": [-38.5, -30.5, 168.5, 180.5],
    "hawaii_big_island": [15.5, 23.5, -161.5, -149.5],
    "japan_hokkaido": [37.5, 45.5, 134.5, 146.5],
    "amazon_forest_west": [-10.0, 0.0, -75.0, -65.0],
    "southeast_asia_maritime": [-5.0, 5.0, 115.0, 125.0],
    "central_africa_congo": [-5.0, 5.0, 15.0, 25.0],
    "greatbarrier_reef_central": [-20.0, -15.0, 145.0, 150.0],
    "amazon_forest_east": [-10.0, 0.0, -55.0, -45.0],
}

O1280_DETAIL_REGIONS: dict[str, list[float]] = {
    "tibet_karakoram_core": [34.5, 38.5, 78.5, 84.5],
    "andes_central_core": [-35.5, -31.5, -72.5, -68.5],
    "greenland_south_fjords": [60.0, 63.5, -49.5, -44.5],
    "horn_of_africa_highlands": [7.0, 11.5, 38.0, 43.5],
    "zagros_core": [32.0, 36.5, 48.0, 54.5],
    "rockies_front_range": [42.0, 46.0, -110.0, -105.0],
    "himalayas_west_core": [32.0, 36.0, 72.0, 78.0],
    "new_zealand_north_core": [-39.0, -35.0, 173.0, 178.0],
    "hawaii_big_island_core": [18.0, 21.5, -157.5, -154.0],
    "hokkaido_core": [41.0, 44.5, 140.0, 145.5],
}

O2560_SHOWCASE_REGIONS: dict[str, list[float]] = {
    "humberto_core": [22.0, 25.0, -60.0, -56.0],
    "java_bali_strait": [-9.0, -6.0, 112.0, 116.0],
    "gbr_reef_tight": [-18.0, -15.0, 146.0, 150.0],
    "amazon_manaus": [-4.0, -1.0, -62.0, -58.0],
    "congo_river_delta": [-7.0, -4.0, 11.0, 15.0],
    "rift_valley_tight": [7.0, 10.0, 37.0, 41.0],
    "alps_innsbruck": [46.0, 49.0, 10.0, 14.0],
    "tokyo_bay": [34.0, 37.0, 138.5, 142.5],
    "florida_keys": [24.0, 27.0, -83.0, -79.0],
    "crete_south": [34.0, 37.0, 23.5, 27.5],
}

MASTER_ONLY_PREDICTION_REGION_BOXES: dict[str, list[float]] = {
    "amazon_forest": [-15.0, 5.0, -75.0, -45.0],
    "eastern_us": [25.0, 45.0, -90.0, -70.0],
    "himalayas": [25.0, 40.0, 75.0, 100.0],
    "southeast_asia": [-10.0, 20.0, 95.0, 150.0],
    "central_africa": [-10.0, 10.0, 10.0, 30.0],
    "idalia": [10.0, 40.0, -100.0, -70.0],
    "idalia_center": [18.0, 32.0, -92.0, -78.0],
    "franklin": [10.0, 40.0, -80.0, -50.0],
    "franklin_center": [18.0, 32.0, -73.0, -59.0],
}

PREDICTION_REGION_BOXES: dict[str, list[float]] = {
    **LEGACY_LOCAL_REGION_BOXES,
    **O96_INTERESTING_REGIONS,
    **O1280_INTERESTING_REGIONS,
    **O1280_DETAIL_REGIONS,
    **O2560_SHOWCASE_REGIONS,
    **MASTER_ONLY_PREDICTION_REGION_BOXES,
}

DEFAULT_MODEL_VARIABLES = ["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"]
DEFAULT_WEATHER_STATES = ["10u", "10v", "2t", "msl", "tp", "z_500", "u_850", "v_850", "t_850"]

LEGACY_O320_PLOTTER_REGIONS = [
    "amazon_forest",
    "european_arctic",
    "himalayas",
    "rocky_mountains",
    "west_sahara",
    "pyrenees_alpes",
    "eastern_us",
    "central_africa",
]

LEGACY_O1280_PLOTTER_REGIONS = [
    "rocky_mountains_central",
    "rocky_mountains_north",
    "rocky_mountains_south",
    "amazon_forest_central",
    "amazon_forest_west",
    "amazon_forest_east",
    "southeast_asia_central",
    "southeast_asia_mainland",
    "southeast_asia_maritime",
    "west_sahara_central",
    "west_sahara_coastal",
    "west_sahara_east",
    "himalayas_central",
    "himalayas_west",
    "himalayas_east",
    "greatbarrier_reef_central",
    "greatbarrier_reef_north",
    "greatbarrier_reef_south",
    "eastern_us_central",
    "eastern_us_north",
    "eastern_us_south",
    "central_africa_congo",
    "central_africa_north",
    "central_africa_south",
]

GRID_CONFIG: dict[str, dict[str, object]] = {
    "O96": {
        "default_region": "amazon_forest",
        "default_suite": list(O96_INTERESTING_REGIONS),
    },
    "O1280": {
        "default_region": "amazon_forest_central",
        "default_suite": list(O1280_INTERESTING_REGIONS),
        "legacy_plotter_regions": LEGACY_O1280_PLOTTER_REGIONS,
        "min_hres_points": 6_000_000,
    },
    "O2560": {
        "default_region": "alps_innsbruck",
        "default_suite": list(O2560_SHOWCASE_REGIONS),
        "min_hres_points": 26_000_000,
    },
    "O320": {
        "default_region": "amazon_forest",
        "legacy_plotter_regions": LEGACY_O320_PLOTTER_REGIONS,
    },
}

RENDER_DPI = 220
