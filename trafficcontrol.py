import cv2
import numpy as np

# Load pre-trained vehicle detection model
vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')

# Initialize video capture
cap = cv2.VideoCapture(0) # Replace 'traffic_video.mp4' with 0 for webcam

def detect_vehicles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 2)
    return vehicles

def draw_traffic_lights(frame, light_status):
    colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0)
    }
    positions = {
        'light1': (50, 50),
        'light2': (50, 100),
        'light3': (50, 150)
    }
    for light, position in positions.items():
        cv2.circle(frame, position, 20, colors[light_status[light]], -1)

light_status = {
    'light1': 'red',
    'light2': 'yellow',
    'light3': 'green'
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles
    vehicles = detect_vehicles(frame)

    # Draw rectangles around detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Draw traffic lights
    draw_traffic_lights(frame, light_status)

    # Update traffic lights based on vehicle count
    if len(vehicles) > 5:
        light_status = {
            'light1': 'green',
            'light2': 'red',
            'light3': 'red'
        }
    else:
        light_status = {
            'light1': 'red',
            'light2': 'green',
            'light3': 'red'
        }

    # Display the frame
    cv2.imshow('Traffic Control', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
