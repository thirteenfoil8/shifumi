import cv2
import mediapipe as mp
from dotenv import dotenv_values
import json
import xgboost as xgb
import pandas as pd
import numpy as np
config = dotenv_values(".env")


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(
                        image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append(cx)
                lmlist.append(cy)
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist

    def get_center(self):
        cap = cv2.VideoCapture(0)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(h, w)

    def gather_data(self, num_samples):
        global rock, paper, scissor
        cap = cv2.VideoCapture(0)
        tracker = handTracker()
        # trigger tells us when to start recording
        trigger = False
        counter = 0

        while True:
            success, image = cap.read()
            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)
            if not success:
                break
            if counter == num_samples:
                trigger = not trigger
                counter = 0

            if trigger:
                if len(lmList) == 42:
                    # Append lm list to the list with the selected class_name
                    eval(class_name).append({counter: lmList})

                    # Increment the counter
                    counter += 1

                    # Text for the counter
                    text = "Collected Samples of {}: {}".format(
                        class_name, counter)
            else:
                text = "Press 'r' to collect rock samples, 'p' for paper, 's' for scissor and 'n' for nothing"

            # Show the counter on the imaege
            cv2.putText(image, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Video", image)
            k = cv2.waitKey(1)
            if k == ord('r'):
                # Trigger the variable inorder to capture the samples
                trigger = not trigger
                class_name = 'rock'
                rock = []
            if k == ord('p'):
                trigger = not trigger
                class_name = 'paper'
                paper = []
            if k == ord('s'):
                trigger = not trigger
                class_name = 'scissor'
                scissor = []
            if k == ord('x'):
                classes = ['rock', 'paper', 'scissor']
                for class_ in classes:
                    file_path = config["DATA_HAND"]
                    with open(file_path + f'{class_}.json', 'w') as file:
                        json.dump(eval(class_), file)
                print("end")

            # Exit if user presses 'q'
            if k == ord('q'):
                break


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    model_xgb = xgb.Booster()
    model_xgb.load_model(config["MODEL_PATH"])

    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = np.array(tracker.positionFinder(image))
        last = None

        if len(lmList) == 42:
            test = xgb.DMatrix(pd.DataFrame(lmList
                                            [..., None].T, columns=list(range(0, 42))))
            new = model_xgb.predict(test)
            print(new)
            print(lmList)

        cv2.imshow("Video", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    # tracker = handTracker()
    # tracker.get_center()
    # tracker.gather_data(10000)
    main()
    print("end")
