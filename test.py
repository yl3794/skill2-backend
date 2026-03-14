import requests
import time

print("Starting angle feed... Press Ctrl+C to stop.")

try:
    while True:
        try:
            response = requests.get("http://localhost:5000/angles", timeout=2)
            angles = response.json()
            print("\n--- Angles Snapshot ---")
            for joint, angle in angles.items():
                print(f"{joint}: {angle}°")
            print("-----------------------")
        except requests.exceptions.ConnectionError:
            print("Backend offline — stopping.")
            break
        except requests.exceptions.Timeout:
            print("Backend not responding — stopping.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

        time.sleep(3)

except KeyboardInterrupt:
    print("\nStopped by user.")