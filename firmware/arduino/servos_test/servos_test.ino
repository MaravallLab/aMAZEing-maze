#include <SerialCommand.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
// you can also call it with a different address you want
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x41);
// you can also call it with a different address and I2C interface

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire);

int servoNum = 100;

// Depending on your servo make, the pulse width min and max may vary, you 
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
// have!
#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600

#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

#define REWSERVOMIN  190 // This is the 'minimum' pulse length count (out of 4096)
#define REWSERVOMAX  560 // This is the 'maximum' pulse length count (out of 4096)
#define REWUSMIN  750 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define REWUSMAX  2250 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600


#define GRATSERVOMIN  225 // This is the 'minimum' pulse length count (out of 4096) for the https://www.pololu.com/file/0J1435/FS90-specs.pdf
#define GRATSERVOMAX  525 // This is the 'maximum' pulse length count (out of 4096) for the https://www.pololu.com/file/0J1435/FS90-specs.pdf
#define GRATSERVOUSMIN 900 // for the https://www.pololu.com/file/0J1435/FS90-specs.pdf
#define GRATSERVOUSMAX 2100 // for the https://www.pololu.com/file/0J1435/FS90-specs.pdf

// our servo # counter
//uint8_t servoNum = 0;

#define gratingLnum 0
#define gratingRnum 1
#define gratingLRnum 2
#define gratingRLnum 3
#define gratingLLnum 4
#define gratingRRnum 5

//reward location A
#define IRsensorA 2
#define rewardMotorA 8
//reward location B
#define IRsensorB 15
#define rewardMotorB 9
//reward location C
#define IRsensorC 17
#define rewardMotorC 7
//reward location D
#define IRsensorD 16
#define rewardMotorD 6

int max_degree =180;

//int servoNum = 0;
int rewardServoMovTime = 400;
int IRsensor = 0;
int degree = 0;
int IRvalue = 0;
int time1 = 0;
int time2 = 0;
int pulseLen = 0;
int pelletDropped = 0;
int rewPulseLen1 = map(90, 0, 180, REWSERVOMIN, REWSERVOMAX);
int rewPulseLen2 = map(45, 0, 180, REWSERVOMIN, REWSERVOMAX);

//SerialCommand sCmd;     // The demo SerialCommand object


void setup() {
  Serial.begin(115200);

  //Wire.begin(14, 27);
  //Wire.begin(5, 18);
  pinMode(IRsensorA,INPUT);
  pinMode(IRsensorB,INPUT);
  pinMode(IRsensorC,INPUT);
  pinMode(IRsensorD,INPUT);



  

  
  // generic functions for development/understanding the parsing library
  //sCmd.addCommand("HELLO", sayHello);        // Echos the string argument back
  //sCmd.addCommand("P",     processCommand);  // Converts two arguments to integers and echos them back
  //sCmd.setDefaultHandler(unrecognized);      // Handler for command that isn't matched  (says "What?")


  //servo library initialization
  pwm.begin();
  /*
   * In theory the internal oscillator (clock) is 25MHz but it really isn't
   * that precise. You can 'calibrate' this by tweaking this number until
   * you get the PWM update frequency you're expecting!
   * The int.osc. for the PCA9685 chip is a range between about 23-27MHz and
   * is used for calculating things like writeMicroseconds()
   * Analog servos run at ~50 Hz updates, It is importaint to use an
   * oscilloscope in setting the int.osc frequency for the I2C PCA9685 chip.
   * 1) Attach the oscilloscope to one of the PWM signal pins and ground on
   *    the I2C PCA9685 chip you are setting the value for.
   * 2) Adjust setOscillatorFrequency() until the PWM update frequency is the
   *    expected value (50Hz for most ESCs)
   * Setting the value here is specific to each individual I2C PCA9685 chip and
   * affects the calculations for the PWM update frequency. 
   * Failure to correctly set the int.osc value will cause unexpected PWM results
   */
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(10);

}// setup

void loop() {
  //Serial.println("aqui");
  for (int i=0;i<10;i++ ){
    for (int servo_num=0;servo_num<10;servo_num++){
      Serial.println(servo_num);
      for (int atempts=0;atempts<5;atempts++){
        if (servo_num<6){
          int serv_min = GRATSERVOMIN;
          int serv_max = GRATSERVOMAX;
          int max_degree =120;
        
        }
        else{
          int serv_min = REWSERVOMIN;
          int serv_max = REWSERVOMAX;
          int max_degree =180;
        }
        for (int degree = 10; degree<max_degree-10;degree = degree+30){
          Serial.println(degree);
        //Serial.println(pulseLen);
        pulseLen = map(degree, 0, max_degree, REWSERVOMIN, REWSERVOMAX);  
        pwm.setPWM(servo_num, 0, pulseLen);
        delay(100);
      }
      for (int degree = max_degree-10; degree>10;degree = degree-30){
        //Serial.println(pulseLen);
        pulseLen = map(degree, 0, max_degree, REWSERVOMIN, REWSERVOMAX);  
        pwm.setPWM(servo_num, 0, pulseLen);
        delay(100);
      }
      }
    }


  }
  //sCmd.readSerial();     // We don't do much, just process serial commands
} //end loop

///servo callback functions -------------------------------------------------/////////////////////////////////////////////
void gratingRoutine(int degree=90, int servoNum=0){


  pulseLen = map(degree, 0, 120, GRATSERVOMIN, GRATSERVOMAX);
  //Serial.println(pulseLen);
  pwm.setPWM(servoNum, 0, pulseLen);

}// grating routines

void pelletRoutine(int IRsensor=0,int servoNum=0, int rewardServoMovTimeLoc = rewardServoMovTime){
  int pelletDropped = 0;
  while (pelletDropped==0){
    pwm.setPWM(servoNum, 0, rewPulseLen1);
    time1=millis();
    time2=millis();
    //Serial.println(IRsensor);
    //Serial.println(rewardServoMovTimeLoc);
    while (time2-time1<rewardServoMovTimeLoc){
      IRvalue = digitalRead(IRsensor);
      //Serial.println(IRvalue);
      if(IRvalue==0){
        pelletDropped=1;
        break;
      }//if
      time2=millis();
    }// while time2-time1
    
    if (pelletDropped==0){
      pwm.setPWM(servoNum, 0, rewPulseLen2);
      time1=millis();
      time2=millis();
      while (time2-time1<rewardServoMovTimeLoc){
        IRvalue = digitalRead(IRsensor);
        //Serial.println(IRvalue);
        if(IRvalue==0){
          pelletDropped=1;
          break;
        }//if
        time2=millis();
      }//while time2-time1
    }// if   
  }//if
}// end pellet routine


