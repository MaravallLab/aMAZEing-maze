const int TRIGGER_PIN = 8; // Connected to TDT BitIn

void setup() {
  Serial.begin(115200);
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == 'H') {       // HIGH (Sound Start)
      digitalWrite(TRIGGER_PIN, HIGH);
    }
    else if (cmd == 'L') {  // LOW (Sound Stop)
      digitalWrite(TRIGGER_PIN, LOW);
    }
  }
}