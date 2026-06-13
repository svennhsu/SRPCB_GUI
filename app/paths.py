from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = ROOT / "cache"
ANNOTATED_CACHE = CACHE_DIR / "annotated"
SR_CACHE = CACHE_DIR / "sr"
SR_PROXY_CACHE = CACHE_DIR / "sr_detection_proxy"
HISTORY_FILE = CACHE_DIR / "history.json"

OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

DETECTION_DIR = ROOT / "detection"
DETECTION_CHECKPOINT = ROOT / "models" / "detection" / "best_model.pth"

SR_DIR = ROOT / "super_resolution"
SR_MODEL_PATH = SR_DIR / "model.py"
SR_CHECKPOINT = ROOT / "models" / "super_resolution" / "best_model.pth"
