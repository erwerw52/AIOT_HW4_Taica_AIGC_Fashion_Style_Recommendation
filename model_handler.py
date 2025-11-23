import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manual mapping based on model card description (Alphabetical order)
ID2LABEL = {
    0: "Blazer",
    1: "Blouse",
    2: "Cardigan",
    3: "Dress",
    4: "Hoodie",
    5: "Jacket",
    6: "Jeans",
    7: "Nightgown",
    8: "Outerwear",
    9: "Pajamas",
    10: "Rain jacket",
    11: "Rain trousers",
    12: "Robe",
    13: "Shirt",
    14: "Shorts",
    15: "Skirt",
    16: "Sweater",
    17: "T-shirt",
    18: "Tank top",
    19: "Tights",
    20: "Top",
    21: "Training top",
    22: "Trousers",
    23: "Tunic",
    24: "Vest",
    25: "Winter jacket",
    26: "Winter trousers"
}

class FashionSystem:
    def __init__(self):
        self.classifier = None
        self.llm_pipeline = None
        self.tokenizer = None
        self.model = None
        
    def load_models(self):
        """
        Loads the classifier and LLM. 
        Note: This can take a significant amount of time and memory.
        """
        # Determine paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_classifier_path = os.path.join(base_path, "model", "classifier")
        local_llm_path = os.path.join(base_path, "model", "llm")

        # Classifier ID or Path
        if os.path.exists(local_classifier_path):
            classifier_model = local_classifier_path
            logger.info(f"Loading Classifier from local path: {classifier_model}")
        else:
            classifier_model = "wargoninnovation/wargon-clothing-classifier"
            logger.info(f"Loading Classifier from Hugging Face: {classifier_model}")

        logger.info("Loading Classifier...")
        try:
            self.classifier = pipeline(
                "image-classification", 
                model=classifier_model
            )
            logger.info("Classifier loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise e

        # LLM ID or Path
        if os.path.exists(local_llm_path):
            model_id = local_llm_path
            logger.info(f"Loading LLM from local path: {model_id}")
        else:
            model_id = "Qwen/Qwen2.5-3B-Instruct"
            logger.info(f"Loading LLM from Hugging Face: {model_id}")

        logger.info(f"Loading LLM ({model_id})...")
        print(f"[System] Loading LLM: {model_id}")
        try:
            # Quantization config for efficiency (requires bitsandbytes)
            # If user doesn't have GPU, this might fail or be very slow.
            # We try to load with 4-bit quantization if CUDA is available.
            if torch.cuda.is_available():
                try:
                    logger.info("Attempting to load with 4-bit quantization...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                    logger.info("4-bit quantization loaded successfully.")
                except Exception as q_err:
                    logger.warning(f"Quantization failed: {q_err}. Falling back to standard GPU load.")
                    print(f"[Warning] Quantization failed ({q_err}). Falling back to standard GPU load.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
            else:
                # CPU fallback (Warning: Very Slow for 7B, but okay for 1.5B)
                logger.warning("CUDA not available. Loading on CPU.")
                print("[Warning] CUDA not available. Loading on CPU.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256
            )
            logger.info("LLM loaded successfully.")
            print("[System] LLM loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            print(f"[Error] Failed to load LLM: {e}")
            # Fallback for demonstration if LLM fails (e.g. no memory)
            logger.warning("Using dummy LLM for demonstration purposes due to load failure.")
            self.llm_pipeline = None

    def classify_image(self, image_path, top_k=None):
        if not self.classifier:
            return [{"label": "Model not loaded", "score": 0.0}]
        
        # Classifier pipeline handles path or PIL image
        # top_k=None returns all classes
        results = self.classifier(image_path, top_k=top_k)
        
        # Map labels to human readable names
        for res in results:
            label_str = res['label']
            # Handle "LABEL_X" format
            if label_str.startswith("LABEL_"):
                try:
                    idx = int(label_str.split("_")[1])
                    if idx in ID2LABEL:
                        res['label'] = ID2LABEL[idx]
                except:
                    pass # Keep original if parsing fails
                    
        return results

    def generate_text(self, prompt):
        if not self.llm_pipeline:
            return "LLM not available. (Mock: Stylish recommendation)"
        
        # Use chat template for better instruction following
        messages = [
            {"role": "system", "content": "You are a helpful fashion assistant. Please always respond in Traditional Chinese (繁體中文)."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        sequences = self.llm_pipeline(
            prompt_formatted,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract only the generated text (remove the prompt)
        generated_text = sequences[0]['generated_text']
        # Qwen/Chat models usually include the prompt in the output, we need to strip it.
        # The simple replace might fail if the template changes slightly, but usually works.
        if generated_text.startswith(prompt_formatted):
            return generated_text[len(prompt_formatted):].strip()
        return generated_text.strip()

# Singleton instance
fashion_system = FashionSystem()
