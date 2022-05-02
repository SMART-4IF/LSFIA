import shutil
import cv2
import time
import numpy as np
import os
import configuration as configuration
import time
import mediapipe as mp

# Paramétrer la session
fps = 30
collection = 'bonjour'
number_video = 30

def record_videos(data_path, action, number_video):

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    increment = 0
    idASCII = 97

    for video in range(number_video):

        print("Record N°" + str(video) + " :: video called " + chr(idASCII) + str(increment)+'.mp4')

        writer = cv2.VideoWriter(data_path + '/' + (idASCII) + str(increment)+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

        while True:
            ret, frame = cap.read()

            writer.write(frame)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        if increment == 9:
            increment = 0
            idASCII += 1
        else:
            increment += 1

        time.sleep(1)  # Sleep for 1 seconds


    cap.release()
    writer.release()
    cv2.destroyAllWindows()

record_videos(configuration.DATA_PATH, collection, number_video)