import Rpi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor():
    def __init__(self, ENA, In1, In2, In3, IN4, ENB):
        self.ENA = ENA
        self.In1 = In1
        self.In2 = In2
        self.In3 = In3
        self.In4 = IN4
        self.ENB = ENB
        GPIO.setup(self.ENA, GPIO.OUT)
        GPIO.setup(self.In1, GPIO.OUT)
        GPIO.setup(self.In2, GPIO.OUT)
        GPIO.setup(self.In3, GPIO.OUT)
        GPIO.setup(self.In4, GPIO.OUT)
        GPIO.setup(self.ENB, GPIO.OUT)
        self.pwmA = GPIO.pwm(self.ENA, 100)
        self.pwmB = GPIO.pwm(self.ENB, 100)
        self.pwmA.start(0)
        self.pwmB.start(0)
    
    def moveForward(self, speed = 50, t=0):
        self.pwmA.ChangeDutyCycle(speed)
        GPIO.output(self.In1,GPIO.LOW)
        GPIO.output(self.In2,GPIO.HIGH)
        self.pwmB.ChangeDutyCycle(speed)
        GPIO.output(self.In3,GPIO.LOW)
        GPIO.output(self.In4,GPIO.HIGH)
        sleep(t)
    
    def stop(self,t=0):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        sleep(t)
        




