import threading
import subprocess

def run_script(name):
    subprocess.run(["python3", name])

# Set up threads
recognition_thread = threading.Thread(target=run_script, args=("gesture_recognition.py",))
visualization_thread = threading.Thread(target=run_script, args=("visualization.py",))

# Start processes on assigned thread
recognition_thread.start()
visualization_thread.start()