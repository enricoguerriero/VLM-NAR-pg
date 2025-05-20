from models.smolvlm import SmolVLM
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

class SmolVLM256(SmolVLM):
    
    def __init__(self, checkpoint: str = None, base_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct", device=None, num_classes: int = 4):
        super().__init__()
        self.model_name = "SmolVLM256"
        
        self.model_name = "SmolVLM"
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model.
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        if checkpoint:
            self.backbone = Idefics3ForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        else:
            self.backbone = Idefics3ForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
        lora_config = LoraConfig(
            r=8,
            lora_alpha=316,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        hidden_size = self.backbone.config.text_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
    
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False