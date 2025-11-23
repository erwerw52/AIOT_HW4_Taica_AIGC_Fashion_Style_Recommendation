import os
import random
from datetime import datetime

LOG_FILE = "log.md"

def log_interaction(role, message):
    """
    Logs an interaction to the log.md file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Escape pipes in message to avoid breaking markdown table
    clean_message = str(message).replace("|", "\|").replace("\n", "<br>")
    
    log_entry = f"| {role} | {clean_message} | {timestamp} |\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

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
