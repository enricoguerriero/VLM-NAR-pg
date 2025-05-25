import torch

def collate_fn_tokens(batch):
    """
    Collate function for DataLoader.
    This function combines a list of dictionaries into a single dictionary
    with concatenated tensors for each key.
    """
    pixel_values_videos = torch.cat([item["pixel_values_videos"] for item in batch], dim=0)
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    
    return {
        "pixel_values_videos": pixel_values_videos,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }