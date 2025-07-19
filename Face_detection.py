import cv2
cap = cv2.VideoCapture(0)

fac= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fac.detectMultiScale(gray, 1.1, 7)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 150, 0), 4)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("v"):
        break

cap.release()
cv2.destroyAllWindows()
