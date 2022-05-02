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

if os.path.exists(os.path.join(configuration.DATA_PATH, collection)):
    shutil.rmtree(os.path.join(configuration.DATA_PATH, collection))
os.makedirs(os.path.join(configuration.DATA_PATH, collection))

def record_videos(data_path, action, number_video):

    print("Record will start in 3sec")

    cap = cv2.VideoCapture(0)

    time.sleep(3)  # Sleep for 1 seconds
    print("Go !")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    number_frame = 40

    increment = 0
    i = 0
    idASCII = 97

    for video in range(number_video):

        print("Record N°" + str(video) + " :: video called " + chr(idASCII) + str(increment)+'.mp4')

        writer = cv2.VideoWriter(os.path.join(data_path+ '/' + str(action), chr(idASCII) + str(increment)+'.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), number_frame, (width, height))

        print("REC **** Souriez")

        while i < number_frame:
            print("i : " + str(i))
            ret, frame = cap.read()

            writer.write(frame)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            i += 1

        print("Record N°" + str(video) + " :: done")

        if increment == 9:
            increment = 0
            idASCII += 1
        else:
            increment += 1

        i = 0

        time.sleep(3)  # Sleep for 1 seconds

    print("Tap ESC to store the videos")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

record_videos(configuration.DATA_PATH, collection, number_video)