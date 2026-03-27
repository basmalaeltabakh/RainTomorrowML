import subprocess
import sys

# Run the actual app from app/main.py
if __name__ == "__main__":
    # Ensure src is in path
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import and run the main app
    from app.main import *
