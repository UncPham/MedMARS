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
DEIM_CHECKPOINT = CHECKPOINT_FOLDER / "best_stg1.pth"
DEIM_CONFIG = "/Users/uncpham/Repo/Medical-Assistant/src/vision_models/DEIM/configs/deim_dfine/deim_hgnetv2_x_vinbigdata_v3_simple.yml"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

LLM_MODEL = os.getenv("LLM_MODEL", "openai")  # Default to 'openai' if not set
