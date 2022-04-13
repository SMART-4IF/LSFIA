import shutil

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilitiesdef mediapipe_detection(image, model):

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data-FRv2')

# Path for import dataset
DATASET_PATH = "..\dataset"

# Actions that we try to detect
# Let actions empty to detect all actions (signs)
# Write down actions wanted like ["action1", "action2", "action3"] (it will manage)
actions_wanted = []
actions = []
action_paths = {}

# Total of videos for each sign
no_sequences = []

# Videos are going to be 30 frames in length
sequence_length = 30

for root, directories, files in os.walk(DATASET_PATH):
    if len(directories) == 0:
        actualdir = root.split("\\")[len(root.split("\\")) - 1]
        if not len(actions_wanted) or actualdir in actions_wanted:
            print(root)
            actions.append(actualdir)
            action_paths[actualdir] = root
            n_seq = 0
            for video in files:
                n_seq += 1
                video_path = os.path.join(DATASET_PATH, actualdir, video)
            no_sequences.append(n_seq)

# Folder start
start_folder = 1


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    #mp_drawing.draw_landmarks(
    #    image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    #)
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def folder_preparation():
    # Génération des dossiers
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    for action in actions:
        if os.path.exists(os.path.join(DATA_PATH, action)):
            shutil.rmtree(os.path.join(DATA_PATH, action))
        os.makedirs(os.path.join(DATA_PATH, action))


    for action, nbVideo in zip(actions, no_sequences):
        if np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int).size != 0:
            dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        else:
            dirmax = 0
        for sequence in range(1, nbVideo + 1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
            except:
                pass


def change_referential(results):
    # Search center point in results
    pose, face, lh, rh = None, None, None, None
    if results.pose_landmarks.landmark:
        ref = results.pose_landmarks.landmark[0]
        pose = np.array(
            [[res.x - ref.x, res.y - ref.y, res.z - ref.z, res.visibility] for res in
             results.pose_landmarks.landmark]
        ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array(
            [[res.x - ref.x, res.y - ref.y, res.z - ref.z] for res in
             results.face_landmarks.landmark]
        ).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array(
            [[res.x - ref.x, res.y - ref.y, res.z - ref.z] for res in
             results.left_hand_landmarks.landmark]
        ).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array(
            [[res.x - ref.x, res.y - ref.y, res.z - ref.z] for res in
             results.right_hand_landmarks.landmark]
        ).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
    else:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
    return np.concatenate([pose, lh, rh]) # face is missing


def analyse_data():
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # NEW LOOP
        # Loop through actions
        for action, nbVideo in zip(actions, no_sequences):

            video_num = 0

            # Loop through sequences aka videos
            for video in os.listdir(action_paths.get(action)):
                cap = cv2.VideoCapture(action_paths.get(action) + "/" + video)

                video_num += 1

                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    success, frame = cap.read()

                    if not success:
                        print("Ignoring empty camera frame on video N° " + str(video_num))
                        break
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    cv2.imshow('OpenCV Feed', image)
                    # cv2.waitKey(2000)
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(video_num), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()

        cap.release()
        cv2.destroyAllWindows()


# Record mediapipe detected sequences of landmarks
def record_data():
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            time.sleep(2)

            # set start_folder
            dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
            start_folder = dirmax - no_sequences + 1

            for sequence in range(start_folder, start_folder + no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = change_referential(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

def frame_count(video_path, manual=False):
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames

def record_dataset():

    setDirectories()
    setEnvironnement()

    DATASET_PATH = 'MAX_DATA-FR'
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # NEW LOOP
        # Loop through actions
        for action, nbVideo in zip(actions, no_sequences):

            video_num = 0

            # Loop through sequences aka videos
            for video in os.listdir(DATASET_PATH + "/" + action):
                cap = cv2.VideoCapture(DATASET_PATH + "/" + action + "/" + video)

                video_num += 1

                #Count number of frame of the video
                nb_frame = frame_count(DATASET_PATH + "/" + action + "/" + video, True)

                # for sequence in range(start_folder, start_folder+no_sequences):
                # Loop through video length aka sequence length
                # while cap.isOpened():
                for frame_num in range(nb_frame):

                    # Read feed
                    success, frame = cap.read()
                    # print("success : " + str(success))

                    if not success:
                        print("Ignoring empty camera frame on video N° " + str(video_num))
                        break
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    cv2.imshow('OpenCV Feed', image)
                    # cv2.waitKey(2000)
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(video_num), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()

        cap.release()
        cv2.destroyAllWindows()

def setDirectories():
    for action, nbVideo in zip(actions, no_sequences):
        if np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int).size != 0:
            dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        else:
            dirmax = 0
        for sequence in range(1, nbVideo + 1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
            except:
                pass

def setEnvironnement():
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MAX_Data-FR')

    # Path for import dataset
    DATASET_PATH = "videos"

    # Actions that we try to detect
    actions = []

    # Total of videos for each sign
    no_sequences = []

    # Videos are going to be 30 frames in length
    sequence_length = 30

    for root, directories, files in os.walk(DATASET_PATH):
        if len(directories) == 0:
            actualdir = root.split("\\")[len(root.split("\\")) - 1]
            actions.append(actualdir)
            actions
            n_seq = 0
            for video in files:
                n_seq += 1
                video_path = DATASET_PATH + "\\" + actualdir + "\\" + video
            no_sequences.append(n_seq)

    # Folder start
    start_folder = 1
