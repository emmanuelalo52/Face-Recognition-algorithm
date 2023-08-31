import cv2

face_cascade = cv2.CascadeClassifier('Haar_cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar_cascade/haarcascade_eye.xml')
glasses_cascade = cv2.CascadeClassifier('Haar_cascade/haarcascade_eye_tree_eyeglasses.xml')
camera = cv2.VideoCapture(0)

while cv2.waitKey(1) == -1:
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.7, 6, minSize=(200, 200))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize=(40, 40))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            glasses = glasses_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(50, 50))

            for (gx, gy, gw, gh) in glasses:
                cv2.rectangle(frame, (x+gx, y+gy), (x+gx+gw, y+gy+gh), (0, 0, 255), 2)

        cv2.imshow('Face Detection', frame)

camera.release()
cv2.destroyAllWindows()