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

#define gratingRRRnum 12
#define gratingLLLnum 13
#define gratingLRRnum 14
#define gratingRLLnum 15

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

SerialCommand sCmd;     // The demo SerialCommand object


void setup() {
  Serial.begin(115200);

  //Wire.begin(14, 27);
  pinMode(IRsensorA,INPUT);
  pinMode(IRsensorB,INPUT);
  pinMode(IRsensorC,INPUT);
  pinMode(IRsensorD,INPUT);



  
  sCmd.addCommand("grtL",  gratingL); // 
  sCmd.addCommand("grtR",  gratingR); // 
  sCmd.addCommand("grtLR",  gratingLR); // 
  sCmd.addCommand("grtRL",  gratingRL); // 
  sCmd.addCommand("grtLL",  gratingLL); // 
  sCmd.addCommand("grtRR",  gratingRR); // 
  sCmd.addCommand("grtRRR",  gratingRRR); //
  sCmd.addCommand("grtLLL",  gratingLLL); //
  sCmd.addCommand("grtLRR",  gratingLRR); // 
  sCmd.addCommand("grtRLL",  gratingRLL); // 
  
  sCmd.addCommand("rewA", rewardA); // activate reward routine (reward 10) // start reward routine on servo 10
  sCmd.addCommand("rewB", rewardB);
  sCmd.addCommand("rewC", rewardC);
  sCmd.addCommand("rewD", rewardD);
  
  // generic functions for development/understanding the parsing library
  sCmd.addCommand("HELLO", sayHello);        // Echos the string argument back
  sCmd.addCommand("P",     processCommand);  // Converts two arguments to integers and echos them back
  sCmd.setDefaultHandler(unrecognized);      // Handler for command that isn't matched  (says "What?")


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
  sCmd.readSerial();     // We don't do much, just process serial commands
} //end loop

// void loop() {
//   sCmd.readSerial();
//   Serial.println("Looping..."); // Add this
//   delay(800); // Add a small delay so it's not too fast
// }

///servo callback functions -------------------------------------------------/////////////////////////////////////////////
void gratingRoutine(int degree=90, int servoNum=0){


  pulseLen = map(degree, 0, 120, GRATSERVOMIN, GRATSERVOMAX);
  Serial.println(pulseLen);
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

// void pelletRoutine(int IRsensor_pin = 0 , int servoNum_motor= 0, int rewardServoMovTimeLoc = rewardServoMovTime) {
//     Serial.print("Reward routine started for motor: ");
//     Serial.print(servoNum_motor);
//     Serial.print(", IR sensor: ");
//     Serial.println(IRsensor_pin);

//     int pelletDropped = 0;
//     unsigned long startTime;

//     while (pelletDropped == 0) {
//         Serial.println("Attempting pellet drop..."); // New print
//         // Try position 1
//         pwm.setPWM(servoNum_motor, 0, rewPulseLen1);
//         startTime = millis();
//         Serial.println("Moving servo to position 1...");

//         while (millis() - startTime < rewardServoMovTimeLoc) {
//             IRvalue = digitalRead(IRsensor_pin);
//             Serial.print("IR value: "); // Keep this for detailed observation
//             Serial.println(IRvalue);    // Keep this for detailed observation
//             if (IRvalue == 0) {
//                 pelletDropped = 1;
//                 Serial.println("Pellet detected! Exiting current attempt."); // New print
//                 break;
//             }
//         }

//         if (pelletDropped == 0) {
//             // If not dropped, try position 2
//             pwm.setPWM(servoNum_motor, 0, rewPulseLen2);
//             startTime = millis();
//             Serial.println("Moving servo to position 2...");

//             while (millis() - startTime < rewardServoMovTimeLoc) {
//                 IRvalue = digitalRead(IRsensor_pin);
//                 Serial.print("IR value: "); // Keep this for detailed observation
//                 Serial.println(IRvalue);    // Keep this for detailed observation
//                 if (IRvalue == 0) {
//                     pelletDropped = 1;
//                     Serial.println("Pellet detected! Exiting current attempt."); // New print
//                     break;
//                 }
//             }
//         }
//         if (pelletDropped == 0) {
//              Serial.println("Pellet not detected after 2 attempts. Retrying..."); // New print
//         }
//         // Add a small delay here if you want to avoid flooding the serial and give the mechanism time
//         // delay(100);
//     }
//     Serial.println("Reward routine finished and exited."); // New print
// }

void gratingL(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingLnum);
  }//if
  else {
    Serial.println("No arguments");
  }   
}//end gratingL

void gratingR(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingRnum);
  }//if
  else {
    Serial.println("No arguments");
  }   
}//end gratingR

void gratingLR(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingLRnum);
  }//if
  else {
    Serial.println("No arguments");
  }   
}//end gratingLR

void gratingRL(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingRLnum);
  }//if
  else {
    Serial.println("No arguments");
  }   
}//end gratingRL

void gratingLL(){
  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingLLnum);
  }//if
  else {
    Serial.println("No arguments");
  }       
}//end gratingLL

void gratingRR(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingRRnum);
  }//if
  else {
    Serial.println("No arguments");
  }     
}//end gratingRR


void gratingLLL(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingLLLnum);
  }//if
  else {
    Serial.println("No arguments");
  }     
}//end gratingLLL

void gratingLRR(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingLRRnum);
  }//if
  else {
    Serial.println("No arguments");
  }     
}//end gratingLRR

void gratingRLL(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingRLLnum);
  }//if
  else {
    Serial.println("No arguments");
  }     
}//end gratingRLL

void gratingRRR(){

  char *arg;
  int degree;
  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  
gratingRoutine(degree=degree, servoNum=gratingRRRnum);
  }//if
  else {
    Serial.println("No arguments");
  }     
}//end gratingRRR













void rewardA(){
  //Serial.println("rewardA");
  pelletRoutine(IRsensor=IRsensorA,servoNum=rewardMotorA);
     
}//end Reward

void rewardB(){
  pelletRoutine(IRsensor=IRsensorB,servoNum=rewardMotorB);
     
}//end Reward
void rewardC(){
  pelletRoutine(IRsensor=IRsensorC,servoNum=rewardMotorC);
     
}//end Reward
void rewardD(){
  pelletRoutine(IRsensor=IRsensorD,servoNum=rewardMotorD);
     
}//end Reward
/////////////////////////////////////////


void test_IR(){


  
}

void sayHello() {

  char *arg;
  arg = sCmd.next();    // Get the next argument from the SerialCommand object buffer
  if (arg != NULL) {    // As long as it existed, take it
    Serial.print("Hello ");
    Serial.println(arg);
  }
  else {
    Serial.println("Hello, whoever you are");
  }
}

void processCommand() {
  int aNumber;
  char *arg;

  Serial.println("We're in processCommand");
  arg = sCmd.next();
  if (arg != NULL) {
    aNumber = atoi(arg);    // Converts a char string to an integer
    Serial.print("First argument was: ");
    Serial.println(aNumber);
  }
  else {
    Serial.println("No arguments");
  }

  arg = sCmd.next();
  if (arg != NULL) {
    aNumber = atol(arg);
    Serial.print("Second argument was: ");
    Serial.println(aNumber);
  }
  else {
    Serial.println("No second argument");
  }
}

// This gets set as the default handler, and gets called when no other command matches.
void unrecognized(const char *command) {
  Serial.println("What?");
}

