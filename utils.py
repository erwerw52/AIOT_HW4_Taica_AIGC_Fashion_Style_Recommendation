import os
import random

def get_random_image(folder_path="simple"):
    """
    Returns a path to a random image from the specified folder.
    """
    if not os.path.exists(folder_path):
        return None
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not images:
        return None
        
    return random.choice(images)
