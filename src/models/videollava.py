import torch
import torch.nn as nn
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from src.models.basemodel import BaseModel
from peft import LoraConfig, get_peft_model, TaskType

class VideoLlava(BaseModel):
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None, num_classes=4):
        super().__init__()
        self.model_name = "VideoLLaVA"
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.backbone = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.backbone = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.backbone.language_model = get_peft_model(
            self.backbone.language_model,
            lora_config
        )

        hidden_size = self.backbone.config.text_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        ).to(self.device)
        # self.classifier = nn.Linear(hidden_size*2, num_classes, bias=True).to(self.device)
        
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False


    def forward(self, pixel_values_videos=None, input_ids=None, attention_mask=None, labels=None, loss_fct=None):
        
        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos.to(self.device),
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            return_dict=True,
            output_hidden_states=True,
        )

        h = outputs.hidden_states[-1]            
        video_token_id = self.backbone.config.video_token_index
        
        video_mask = (input_ids == video_token_id)
        
        pooled_video = (h * video_mask.unsqueeze(-1)).sum(1) / \
               video_mask.sum(1, keepdim=True).clamp(min=1)
        
        cls_text = h[:, 0, :]  # context (to check if keep it or not)
        fused = torch.cat([pooled_video, cls_text], dim=-1)
        # if doing this, need to change the classifier to accept 2 * hidden_size
        # fused = pooled_video
        
        logits    = self.classifier(fused.float())
        
        if labels is not None and loss_fct is not None:
            loss = loss_fct(logits, labels.float())
        else:
            loss = None
        
        return {"loss": loss, "logits": logits}

    def prompt_definition(self, question: str, system_message: str = "You are a helpful assistant."):
        """
        Build the prompt text for a given question.
        Here, we follow the recommended prompt format for Video LLaVA.
        """
        prompt = f"USER: {system_message}\n<video>\n{question}\nASSISTANT:"
        
        return prompt
    
    def process_input(self, video: list, prompt: str, system_message: str):
        
        final_prompt = self.prompt_definition(prompt, system_message)
        inputs = self.processor(
            text = final_prompt,
            videos = video,
            padding = True,
            do_rescale = False,
            return_tensors = "pt")
        
        return inputs
    
    
    def set_freezing_condition(self, mode: str):
        # 1) freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # 2) unfreeze based on mode
        if mode == "none":
            # full fine-tuning: unfreeze every param
            for param in self.parameters():
                param.requires_grad = True

        elif mode == "all":
            # only the classification head
            for param in self.classifier.parameters():
                param.requires_grad = True

        elif mode == "lora":
            # only LoRA adapters + classification head
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown freezing mode: {mode!r}")

        return False
    
    def unfreeze_schedule(self, x):
        pass
    
    def collate_fn_tokens(self, batch):
        """
        Collate function for DataLoader.
        """
        pixel_values_videos = torch.cat([item["pixel_values_videos"] for item in batch], dim=0)
        input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        return {"pixel_values_videos": pixel_values_videos,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}
    
    
    
    # ---------- From features training methods ---------- #
    
    def forward_classifier(self, features, labels, loss_fct=None):
        """
        Forward pass through the classifier.
        """
        logits = self.classifier(features.float())
        
        if loss_fct is not None:
            loss = loss_fct(logits, labels.float())
        else:
            loss = None
        
        return {"loss": loss, "logits": logits}
    
    def get_input(self, batch):
        """
        Get the input tensors from the batch.
        """
        for key in batch:
            batch[key] = batch[key].squeeze(0).to(self.device)
        return batch
    
    def feature_extraction(self, pixel_values_videos=None, input_ids=None, attention_mask=None):
        """
        Extract features from the model.
        """
        # print(f"pixel_values_videos: {pixel_values_videos}", flush=True)
        # print(f"input_ids: {input_ids}", flush=True)
        # print(f"attention_mask: {attention_mask}", flush=True)
        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # last_layer = outputs.hidden_states[-1] # (batch, seq_len, hidden_dim)
        # pooled = last_layer.mean(dim=1)  # CLS token representation
        
        h = outputs.hidden_states[-1]            
        video_token_id = self.backbone.config.video_token_index
        
        video_mask = (input_ids == video_token_id)
        
        pooled = (h * video_mask.unsqueeze(-1)).sum(1) / \
               video_mask.sum(1, keepdim=True).clamp(min=1)
        cls_text = h[:, 0, :]  # context (to check if keep it or not)
        fused = torch.cat([pooled, cls_text], dim=-1)
        
        
        return fused.float()
 
    @torch.no_grad()
    def generate_answer(
        self,
        inputs,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> str:
        """
        Traditional generative usage of Video-LLaVA (zero-shot).

        Args:
            video: list of frames or a video file path the processor can read.
            question: the user question about the clip.
            system_message: high-level system instruction to prepend.
            max_new_tokens: length of the answer to generate.
            **generate_kwargs: any other `generate` kwargs (e.g. temperature, top_p).

        Returns:
            A single decoded answer string.
        """

        # Move everything to the right device / dtype
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        self.backbone.eval()
        generated_ids = self.backbone.generate(
            pixel_values_videos=inputs["pixel_values_videos"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,              
            do_sample=True     
        )

        # Decode
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # The answer often comes after the original prompt – strip it if desired
        return answer.split("ASSISTANT:")[-1].strip()