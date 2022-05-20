import RPi.GPIO as GPIO

from time import sleep

pin = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin,GPIO.OUT)

pwm=GPIO.PWM(pin,50) #50hz is fine, don't change this
pwm.start(0)

for i in range(1):
    pwm.ChangeDutyCycle(4.5)
    sleep(2)
    print("left")
    pwm.ChangeDutyCycle(9.5)
    sleep(2)
    print("right")
    pwm.ChangeDutyCycle(7)
    sleep(2)
    print("middle")

pwm.stop()

