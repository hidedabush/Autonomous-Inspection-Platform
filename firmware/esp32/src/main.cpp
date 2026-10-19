// AIP station firmware – Phase 1 placeholder.
// Implements the serial protocol in docs/interfaces.md: sends TRIG, accepts PASS/REJECT/PING.
#include <Arduino.h>

const int SENSOR_PIN = 4;   // TODO: confirm wiring
const int LED_PIN = 2;
unsigned long partId = 0;
bool lastState = HIGH;

void setup() {
    Serial.begin(115200);
    pinMode(SENSOR_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    Serial.println("AIP firmware ready");
}

void loop() {
    bool state = digitalRead(SENSOR_PIN);
    if (lastState == HIGH && state == LOW) {  // beam broken
        delay(20);                             // simple debounce
        if (digitalRead(SENSOR_PIN) == LOW) {
            Serial.print("TRIG ");
            Serial.println(++partId);
        }
    }
    lastState = state;

    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "PING") {
            Serial.println("ACK PING");
        } else if (cmd == "PASS" || cmd == "REJECT") {
            digitalWrite(LED_PIN, cmd == "REJECT" ? HIGH : LOW);  // servo control added in Phase 2
            Serial.println("ACK " + cmd);
        } else if (cmd.length() > 0) {
            Serial.println("ERR unknown command");
        }
    }
}
