import cv2
import model as m
import datacollection as datacollection
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


def test_model(X_test, y_test):
    yhat = m.model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    accuracy_score(ytrue, yhat)


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 100, 16), (117, 25, 16), (36, 117, 245), (245, 117, 56), (127, 245, 16), (16, 117, 245), (56, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


def realtime_prediction():
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with datacollection.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = datacollection.mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            datacollection.draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = datacollection.change_referential(results)  # TODO extract key points with referential change
            sequence.append(keypoints)
            sequence = sequence[-1:]

            if len(sequence) == 1:
                res = m.model.predict(np.expand_dims(sequence, axis=0))[0]
                print("Res = " + str(res))
                best_fit = np.argmax(res)
                print('Label = ' + datacollection.actions[best_fit] + ' accuracy = ' + str(best_fit))
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if datacollection.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(datacollection.actions[np.argmax(res)])
                        else:
                            sentence.append(datacollection.actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, datacollection.actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()