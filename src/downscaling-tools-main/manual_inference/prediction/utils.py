from __future__ import annotations


def extract_filtered_input_from_output(
    input_weather_states,
    input_name_to_index,
    output_name_to_index,
):
    common_weather_states = set(input_name_to_index.keys()) & set(
        output_name_to_index.keys()
    )
    filtered_keys_sorted = sorted(
        common_weather_states, key=lambda k: output_name_to_index[k]
    )
    filtered_indices_sorted = [input_name_to_index[key] for key in filtered_keys_sorted]
    filtered_input_weather_states = input_weather_states[..., filtered_indices_sorted]
    filtered_input_name_to_index = {
        key: i for i, key in enumerate(filtered_keys_sorted)
    }
    return filtered_input_weather_states, filtered_input_name_to_index
