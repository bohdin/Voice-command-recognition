import numpy as np

from tensorflow.keras.models import load_model

from Helpers.recording_helper import record_audio, terminate
from Helpers.tf_helper import preprocess_audiobuffer
from Helpers.turtle_helper import move_turtle

commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

model = load_model('Models/final_model.keras')

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print(f'Predicted label: {command}')
    return command

if __name__ == '__main__':
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break