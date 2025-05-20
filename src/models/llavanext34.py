from models.llavanext import LlavaNext
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


class LlavaNext34(LlavaNext):
    """
    A wrapper for the LlavaNext model that encapsulates loading the processor and model,
    as well as generating answers from prompts.
    """
    def __init__(self, 
                 checkpoint: str = None, 
                 base_model_id: str = "llava-hf/LLaVA-NeXT-Video-34B-hf", 
                 device=None, 
                 num_classes=4,
                 lora_modality = "language"):
        super().__init__()
        
        self.model_name = "LLavaNext34B"
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if checkpoint:
            self.backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        if lora_modality == "language":
            self.backbone.language_model = get_peft_model(self.backbone.language_model, 
                                           lora_config)
        elif lora_modality == "total":
            self.backbone = get_peft_model(self.backbone, 
                                           lora_config)
        
        hidden_size = self.backbone.config.text_config.hidden_size
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # ).to(self.device)
        self.classifier = nn.Linear(hidden_size*2, num_classes, bias=True).to(self.device)
        
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        