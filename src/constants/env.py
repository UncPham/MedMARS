from dotenv import load_dotenv
import os
from pathlib import Path

# Get project root directory (parent of src folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent

load_dotenv()

# Static folder for storing images
STATIC_FOLDER = PROJECT_ROOT / "src" / "static"

# Checkpoint folder for model checkpoints
CHECKPOINT_FOLDER = PROJECT_ROOT / "src" / "checkpoint"

# Ensure directories exist
STATIC_FOLDER.mkdir(exist_ok=True)

# Specific checkpoint files
GROUNDING_DINO_CHECKPOINT = CHECKPOINT_FOLDER / "groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = CHECKPOINT_FOLDER / "sam_vit_h_4b8939.pth"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


