# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)

# '''
# for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
# for local webcam use cv2.VideoCapture(0)
# '''


from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit

import numpy as np

MODEL_PATH = "model_v1.pkl"

# FLASK SERVER
# --------------------------------------------

port = 5000
app = Flask(__name__)
# @app.route('/flask', methods=['GET','POST'])

app.config['SECRET_KEY'] = 'this is my secret!'
socketio = SocketIO(app)

@socketio.on('my event')
def handle_my_event(json):
    print('received message: ' + str(json))
    emit('my_response',{'data': 'Connected!'})

# --------------------------------------------

if __name__ == "__main__":

    print("running flask server")
    app.run(port=port, debug=True, threaded=True) # threaded=True is for serving multiple clients

    

