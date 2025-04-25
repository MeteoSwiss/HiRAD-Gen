import datetime
from hirad.datasets import init_dataset_from_config
from hirad.utils.function_utils import convert_datetime_to_cftime


def get_dataset_and_sampler(dataset_cfg, times, has_lead_time=False):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    if has_lead_time:
        plot_times = times
    else:
        plot_times = [
            convert_datetime_to_cftime(
                datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            )
            for time in times
        ]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler