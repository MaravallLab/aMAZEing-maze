# Firmware

Sketches for the microcontrollers, in the `firmware/` folder of the repository.

the code to run the servos requires an adafruit library as well as a serial communication parsing library. they can be found:

- https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library
- https://github.com/kroimon/Arduino-SerialCommand

more info on how to use the adafruit library here: 
https://learn.adafruit.com/16-channel-pwm-servo-driver/using-the-adafruit-library


Serial commands are structured in the following way:

- to change the orientation of a grating, the command going to the arduino board should be a string with the keyword "grating", followed by the coded location of the grating and the angle the grating needs to be moved to. Example: "gratingLL 180"

- to send commands for rewards to be dispensed, the command going to the arduino board should be a string with the keyword "reward", folllowed by the location where the reward is to be given. Example: "rewardA"


## TTL synchronisation

The `firmware/ttl_bnc/TTL_bnc.ino` sketch listens on the serial port and drives one pin: `H` takes it high when a sound starts, `L` takes it low when the sound ends or the animal leaves the arm. Wire that pin to the digital input of the recording system you want to align to.

## MicroPython

`firmware/micropython/` holds the Adafruit PCA9685 driver for running the servos from a MicroPython board instead of an Arduino.
