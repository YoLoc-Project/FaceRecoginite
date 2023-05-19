import os
import requests
from picamera import PiCamera
from parinya import LINE
# Configure the Line notification API token and URL

# Line Notify API URL
LINE_NOTIFY_API_URL = "https://notify-api.line.me/api/notify"

# Your Line token
LINE_TOKEN = "2ZJU8ttq67ClPh5wLyh43TkwxDklGxGmgOl8Jnxgdx0"
line = LINE('2ZJU8ttq67ClPh5wLyh43TkwxDklGxGmgOl8Jnxgdx0')
# Initialize the camera
camera = PiCamera()

# Set the resolution of the camera
camera.resolution = (640, 480)

# Capture an image
image_name = 'image.jpg'
camera.capture(image_name)

# Create the request headers with the Line token
headers = {
    "Authorization": "Bearer " + LINE_TOKEN,
    "Content-Type": "application/x-www-form-urlencoded"
}

# Create the request data with the image file
data = {
    'message': {
        'type': 'image',
        'originalContentUrl': 'https://example.com/' + image_name,
        'previewImageUrl': 'https://example.com/' + image_name
    }
}

# Send the Line notification
response = requests.post(LINE_NOTIFY_API_URL, headers=headers, json=data)

# Delete the captured image
os.remove(image_name)
