import WebcamModuleLabeled as wM
import DataCollectionModule as dcM
import JoyStickModule as jsM
import MotorModule as mM
import cv2
from time import sleep


maxThrottle = 0.45
motor = mM.Motor(2, 3, 4, 17, 22, 27)

record = 0
while True:
    joyVal = jsM.getJS()
    #print(joyVal)
    steering = joyVal['axis1']
    throttle = joyVal['o']*maxThrottle
    backwards = joyVal['x']*maxThrottle
    if joyVal['share'] == 1:
        if record ==0: print('Screen shared ...')
        record +=1
        sleep(0.300)
    if record == 1:
       	img = wM.getImg(True,size=[500,300])
    elif record == 2:
        record = 0
    if joyVal['x']:
        motor.move(-backwards,-steering)
    else:
        motor.move(throttle,-steering)
    cv2.waitKey(1)
