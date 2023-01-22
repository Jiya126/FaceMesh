import cv2
import mediapipe as mp

MpFaceMesh = mp.solutions.face_mesh
faceMesh = MpFaceMesh.FaceMesh()
draw = mp.solutions.drawing_utils
drawSpec = draw.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while True:
    s, img = cap.read()
    imgC = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = faceMesh.process(imgC)
    
    if out.multi_face_landmarks:
        for fLms in out.multi_face_landmarks:
            draw.draw_landmarks(img, fLms, MpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
        for id, lm in enumerate(fLms.landmark):
            ih, iw, ic = img.shape
            x,y = int(lm.x * iw), int(lm.y * ih)
            print(id,x,y)


    cv2.imshow('live', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()