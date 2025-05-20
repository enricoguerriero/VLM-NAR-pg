import yaml

def load_config(model_name: str):
    config_path = f"configs/{model_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)