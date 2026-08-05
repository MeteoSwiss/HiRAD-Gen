REAL_TO_ERA_CHANNEL_MAP = {
    'T_2M': '2t',
    'U_10M': '10u',
    'V_10M': '10v',
    'TOT_PREC_1H': 'tp'
}

REAL_TO_ERA_CHANNEL_MAP_6H = {
    'T_2M': '2t',
    'U_10M': '10u',
    'V_10M': '10v',
    'TOT_PREC_6H': 'tp'
}

ERA_TO_REAL_CHANNEL_MAP = {v: k for k, v in REAL_TO_ERA_CHANNEL_MAP.items()}

