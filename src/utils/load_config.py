import yaml
from collections.abc import Mapping

def deep_merge(dict1, dict2):
    """Recursively merges dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(model_name: str):
    base_config_path = "configs/base.yaml"
    specific_config_path = f"configs/{model_name}.yaml"

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f) or {}

    with open(specific_config_path, "r") as f:
        specific_config = yaml.safe_load(f) or {}

    final_config = deep_merge(base_config, specific_config)
    return final_config
