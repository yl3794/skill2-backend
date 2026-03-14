import threading
import time
import requests
from app import app, cap


def angle_printer():
    print("⏳ Waiting for Flask to start...")
    time.sleep(2)
    print("📊 Angle feed started — printing every 3 seconds\n")

    while True:
        try:
            response = requests.get("http://localhost:5000/angles", timeout=2)
            angles = response.json()

            if angles:
                print("\n--- Joint Angles ---")
                for joint, angle in angles.items():
                    print(f"  {joint}: {angle}°")
                print("--------------------")
            else:
                print("⏳ Waiting for pose to be detected...")

        except requests.exceptions.ConnectionError:
            print("❌ Flask offline — stopping.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

        time.sleep(3)

# Initializes all the files  
if __name__ == '__main__':
    print("=" * 45)
    print("  🏋️  FormCoach Starting...")
    print("=" * 45)
    print("  📡 Flask  → http://localhost:5000")
    print("  🎥 Video  → http://localhost:5000/video")
    print("  📐 Angles → http://localhost:5000/angles")
    print("  💬 Coach  → http://localhost:5000/feedback")
    print("=" * 45)
    print("  Press Ctrl+C to stop everything\n")

    # Start angle printer in background
    threading.Thread(target=angle_printer, daemon=True).start()

    # Start Flask — blocks here until Ctrl+C
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        cap.release()
        print("✅ Done!")
