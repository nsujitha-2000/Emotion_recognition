import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras_preprocessing.image import img_to_array


model = model_from_json(open("pred_model.json", "r").read())
model.load_weights('model.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

x = 1


def image_read(image_path):
    path_check = os.path.exists(image_path)
    print(path_check)
    if path_check:
        # print("Inside path check TRUE")
        image = cv2.imread(image_path)
    return image


while x != 0:
    print("Press 0 to exit")
    print("Enter the path of the image:")

    val = input()
    if val == '0':
        x = 0
    else:
        path = val.replace('"', '')
        image = image_read(path)
        init_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_detected = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(converted_image, (x, y), (x + w, y + h), (255, 0, 0))
            roi_gray = image[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            # image_pixels = img_to_array(roi_gray)
            # image_pixels = np.expand_dims(image_pixels, axis=0)
            # image_pixels /= 255
            image_pixels = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            image_pixels = image_pixels.reshape(-1, 48, 48, 1)
            image_pixels = np.array(image_pixels)
            image_pixels = image_pixels.astype('float32')
            image_pixels /= 255

            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])

            emotion_detection = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
            emotion_prediction = emotion_detection[max_index]

            cv2.putText(image, emotion_prediction, (250, 250), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

            print(emotion_prediction)
            # cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            resized_image = cv2.resize(image, (700, 700))
            cv2.imshow('Emotion', resized_image)
            if cv2.waitKey(10) == ord('b'):
                break

# cap.release()
cv2.destroyAllWindows
