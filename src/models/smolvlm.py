from transformers import AutoProcessor
from transformers import Idefics3ForConditionalGeneration
from models.basemodel import BaseModel
import torch
import re
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

class SmolVLM(BaseModel):
    """
    A wrapper for the smolvlm model that encapsulates loading the processor and model,
    as well as generating answers from prompts.
    """
    def __init__(self, checkpoint: str = None, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device=None, num_classes: int = 4):
        super().__init__()
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
        
    def forward(self, pixel_values, input_ids, attention_mask, labels=None, loss_fct=None):
        """
        Forward pass through the model.
        """
        
        outputs = self.backbone(
            pixel_values = pixel_values.to(self.device),
            input_ids = input_ids.to(self.device),
            attention_mask = attention_mask.to(self.device),
            return_dict=True,
            output_hidden_states=True
        )
        last_layer = outputs.hidden_states[-1]
        pooled = last_layer[:, 0, :]
        logits = self.classifier(pooled.float())
        
        if labels is not None and loss_fct is not None:
            loss = loss_fct(logits, labels)
        else:
            loss = None
        return {"loss": loss, "logits": logits}
    
    def prompt_definition(self, video: list, prompt: str, system_message: str = None):
        """
        Prepares the input for the model.
        """
        image_tokens = [{"type": "image"} for _ in range(len(video))]
        
        prompt_template = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_message},
                ]
            },
            {
                "role": "user",
                "content": [
                    *image_tokens,
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(prompt_template, add_generation_prompt=True)
        
        return prompt
    
    def process_input(self, video: list, prompt: str, system_message: str = None):
        """
        Processes the input video and prompt.
        """
        final_prompt = self.prompt_definition(video, prompt, system_message)
        
        return self.processor(
            text=final_prompt,
            images=video,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
    
    def set_freezing_condition(self, mode: str):
        """
        Sets the cleaning condition for the model.
        """
        for param in self.parameters():
            param.requires_grad = False
        
        if mode == "none":
            for name, param in self.parameters():
                param.requires_grad = True
        
        elif mode == "all":
            for name, param in self.classifier.parameters():
                param.requires_grad = True
                
        elif mode == "lora":
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
                
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'none', 'all', or 'lora'.")
        
    def unfreeze_schedule(self, x):
        pass
    
    def collate_fn_tokens(self, batch):
        """
        Collate function for the model.
        """
        return {
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
            "input_ids": torch.cat([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.cat([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.stack([item["labels"] for item in batch], dim=0),
        }