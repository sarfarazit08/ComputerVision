from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUiType

import sys, cv2, winsound

ui,_= loadUiType("webcam-desktop-app.ui")

class MainApp(QMainWindow, ui):
    volume = 100
    def __init__(self) -> None:
        QMainWindow.__init__(self)
        self.setupUi(self)

        # Write Button Events Bindings
        self.Monitoring.clicked.connect(self.start_monitoring)
        self.Volume.clicked.connect(self.set_volume)
        self.Exit.clicked.connect(self.exit_window)
        self.volumeSlider.setVisible(False)
        self.volumeLevel.setVisible(False)
        self.volumeSlider.valueChanged.connect(self.set_volume_level)
          
    def start_monitoring(self):
        webcam = cv2.VideoCapture(0)
        while True:
            _,frame1 =webcam.read()
            _,frame2 =webcam.read()

            # Calculate frame differences
            diff = cv2.absdiff(frame1,frame2)
            grayFrame = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurFrame = cv2.GaussianBlur(grayFrame,(5,5),0)
            _,threshold = cv2.threshold(blurFrame, 20, 255, cv2.THRESH_BINARY)
            dialated = cv2.dilate(threshold,None,iterations=3)
            contours,_ = cv2.findContours(dialated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 5000:
                    continue
                x, y , w, h = cv2.boundingRect(c)
                cv2.rectangle(frame1,(x,y),(x+w,y+h), (0,0,255),2)
                cv2.imwrite("./captured.jpg", frame1)
                image = QImage("./captured.jpg")
                pm = QPixmap.fromImage(image)
                self.Screen.setPixmap(pm)
                winsound.Beep(self.volume, 100)
                 
            cv2.imshow("WebCam Feed", frame1)
            key = cv2.waitKey(10)
            if key == 27: # Press Escape Key to Exit
                break
        webcam.release()
        cv2.destroyAllWindows()

    def set_volume(self):
        self.volumeSlider.setVisible(True)

    def set_volume_level(self):
        self.volumeLevel.setVisible(True) 
        self.volumeLevel.setText(str(self.volumeSlider.value()))
        self.volume = self.volumeSlider.value()

    def exit_window(self):
        self.close()

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
