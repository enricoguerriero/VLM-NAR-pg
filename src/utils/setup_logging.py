import logging

def setup_logging(model_name: str, task: str):
    """
    Set up logging for the model training and evaluation.
    """
    logger = logging.getLogger(f"{model_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(f"logs/{model_name}_{task}.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger