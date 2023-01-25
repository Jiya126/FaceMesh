import cv2
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self, staticMode, maxFaces, minDetectionConf, minTrackConf):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionConf = minDetectionConf
        self.minTrackConf = minTrackConf

        self.MpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.MpFaceMesh.FaceMesh()
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw):
        self.imgC = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.out = self.faceMesh.process(self.imgC)
        faceCoord = []
        faces = []
    
        if self.out.multi_face_landmarks:
            for fLms in self.out.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, fLms, self.MpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
            for id, lm in enumerate(fLms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x * iw), int(lm.y * ih)
                # print(id,x,y)
            faceCoord.append([x,y])
            faces.append(faceCoord)

        return faces, img


def main():
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector(staticMode=False, maxFaces=5, minDetectionConf=0.5, minTrackConf=0.5)

    while True:
        s, img = cap.read()
        faces, img = detector.findFaceMesh(img, draw=True)
        # print(faces)
        cv2.imshow('live', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()

    # cap.release()
    # cv2.destroyAllWindows()