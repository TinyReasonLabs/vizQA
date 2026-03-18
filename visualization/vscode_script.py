import os
import time
import subprocess

# Watch the current directory
WATCH_PATH = "."


def get_latest_image():
    files = [f for f in os.listdir(WATCH_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not files: return None
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(WATCH_PATH, x)), reverse=True)
    return files[0]

last_seen = None
print(f"Monitoring {os.path.abspath(WATCH_PATH)} for new images...")

while True:
    latest = get_latest_image()
    if latest and latest != last_seen:
        # Use the 'code' command to open the image in the current VS Code instance
        # -r reuse the last active window
        print(f"Opening {os.path.join(WATCH_PATH, latest)} in VS Code...")
        subprocess.run(["code", f"{os.path.join(WATCH_PATH, latest)}"],shell=True)
        last_seen = latest
    time.sleep(0.1) # Check every second
