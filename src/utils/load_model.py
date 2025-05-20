import os

def load_model(model_name: str, checkpoint: str):
    """
    Load the model based on the model name and checkpoint path.
    
    Args:
        model_name (str): The name of the model to load.
        checkpoint (str): The path to the model checkpoint.
        
    Returns:
        model: The loaded model.
    """
    if checkpoint is not None and not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file {checkpoint} does not exist.")
    
    if model_name == "VideoLLaVA":
        from src.models.videollava import VideoLlava
        model = VideoLlava(checkpoint_path=checkpoint)
    elif model_name == "SmolVLM":
        from src.models.smolvlm import SmolVLM
        model = SmolVLM(checkpoint=checkpoint)
    elif model_name == "SmolVLM256":
        from src.models.smolvlm256 import SmolVLM256
        model = SmolVLM256(checkpoint=checkpoint)
    elif model_name == "TimeSformer":
        from src.models.timesformer import TimeSformer
        model = TimeSformer()
    elif model_name == "LLaVANeXT":
        from src.models.llavanext import LlavaNext
        model = LlavaNext(checkpoint=checkpoint)
    elif model_name == "LLaVANeXT34":
        from src.models.llavanext34 import LlavaNext34
        model = LlavaNext34(checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
