import serial
import time
import cv2 as cv
import sounddevice as sd
from typing import Optional, Tuple
from config import ExperimentConfig

#Class Arduino, define the commands for the TTL as functions (serial.write "H" to determine the onset of the TTL, serial.write "L" to determine the offset of the TTL)

class ArduinoController: 

    def __init__(self, cfg: ExperimentConfig, active: bool= True):
        self.ser: Optional[serial.Serial] = None # stores the serial connection with Arduino 
        self.active = active
        self.port = cfg.arduino_port
        self.baud_rate = cfg.arduino_baud

        if self.active: 
            try:
                self.ser = serial.Serial(self.port, self.baud_rate, timeout= 0.1)
                time.sleep(2)
                print(f"Arduino connected (State Machine Mode) on {self.port}")
            except Exception as e:
                raise RuntimeError(f"Critical error, Arduino connection on port {self.port} failed.\nDetails{e}")
            
    def trigger_on(self):
        # signal the TTL onset
        if self.active and self.ser: 
            self.ser.write(b'H')

    def trigger_off(self):
        #TTL offset 
        if self.active and self.ser:
            self.ser.write(b'L')

    def close(self):
        # close the connection to Ardu
        if self.ser: 
            self.ser.close() 


#Class for video

# here we have all the functions related to video capturing 
class Camera:
    def __init__(self, device_id: int= 0):
        self.cap = cv.VideoCapture(device_id)

        if not self.cap.isOpened():
            #stop everything if the camera is missing
            raise RuntimeError(f"Error: Cannot open camera")
        
        #get camera properties

        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))

        print(f"camera initialised: {self.width}x{self.height} @ {self.fps} fps")

    def get_frame(self):
        #tries to read a single frame, returns (siccess, frame)

        valid, frame = self.cap.read()

        if not valid:
            return False, None
        return True, frame
    

    def release(self):
        return self.cap.release()


