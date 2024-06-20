import cv2
import mediapipe as mp
import utils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# For webcam input:
def generate_stream(source=0, live=True):
    cap = cv2.VideoCapture(source)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                if live: continue
                else: break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
            
                # yield results.multi_hand_landmarks[0].landmark
                yield results

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            # Press esc to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()



if __name__ == "__main__":
    # for landmark in generate_stream("video\A.webm", False):
    #     print(landmark)
    import predictor
    # predictor.load_model(predictor.EUCLIDEAN_MODEL)
    # for landmark in generate_stream():
    #     df = utils.landmark2df(landmark)
    #     letter, conf = predictor.predict_with_conf(df)
    #     if conf > 0.5:
    #         print(letter, f"{int(conf*100)}%")
    model = predictor.Predictor("modelSIBI.h5")
    for result in generate_stream(rf"D:\Python\RPL-AI\sibi\A\A (5).jpg", False):
        lmks = utils.Landmark.from_legacy_mp_result(result)
        print(lmks.to_json())
        # char, conf = model.predict(result)
        # print(char, f"{conf*100:}%")