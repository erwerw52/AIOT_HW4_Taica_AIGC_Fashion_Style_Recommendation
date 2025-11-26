import torch
from transformers import pipeline
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Load environment variables
load_dotenv()

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
        self.client = None
        self.init_error = None
        
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
        
        # Try loading with token first (if available), then without, then fallback
        hf_token = os.getenv("HF_TOKEN")
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Attempt 1: Use token if available
            kwargs = {"device": device}
            if hf_token:
                kwargs["token"] = hf_token
            
            logger.info(f"Attempting to load classifier with token configuration...")
            self.classifier = pipeline("image-classification", model=classifier_model, **kwargs)
            logger.info("Classifier loaded successfully.")
            
        except Exception as e1:
            logger.warning(f"First attempt to load classifier failed: {e1}")
            
            try:
                # Attempt 2: Try without token (force public access)
                logger.info("Attempting to load classifier without token...")
                self.classifier = pipeline("image-classification", model=classifier_model, device=device, token=False)
                logger.info("Classifier loaded successfully (without token).")
                
            except Exception as e2:
                logger.error(f"Second attempt failed: {e2}")
                
                # Attempt 3: Fallback model
                fallback_model = "google/vit-base-patch16-224"
                logger.info(f"Attempting fallback to {fallback_model}...")
                try:
                    self.classifier = pipeline("image-classification", model=fallback_model, device=device)
                    logger.info("Fallback classifier loaded successfully.")
                except Exception as e3:
                    logger.error(f"All classifier loading attempts failed. Last error: {e3}")
                    self.classifier = None
                    # Don't raise here, allow app to start without classifier (will show error in UI)
                    self.init_error = f"Classifier load failed: {e1}"

        # LLM Setup via API
        # Switched to Llama 3 as requested
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        # Try to get token from env
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            logger.warning("HF_TOKEN not found in environment variables. Please check your .env file.")
            print("[Warning] HF_TOKEN not found. LLM features might fail.")

        logger.info(f"Initializing InferenceClient for {model_id}...")
        print(f"[System] Initializing InferenceClient: {model_id}")
        
        try:
            self.client = InferenceClient(model=model_id, token=hf_token)
            logger.info("InferenceClient initialized successfully.")
            print("[System] InferenceClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize InferenceClient: {e}")
            print(f"[Error] Failed to initialize InferenceClient: {e}")
            self.client = None
            self.init_error = str(e)

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

    def generate_text(self, prompt, model_id=None):
        if not self.client:
            return "LLM API not available. (Mock: Stylish recommendation)"
        
        messages = [
            {"role": "system", "content": "You are a helpful fashion assistant. Please always respond in Traditional Chinese (繁體中文)."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Prepare arguments
            kwargs = {
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            }
            # If a specific model is requested, override the default
            if model_id:
                kwargs["model"] = model_id

            response = self.client.chat_completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating text: {e}"

# Singleton instance
fashion_system = FashionSystem()
