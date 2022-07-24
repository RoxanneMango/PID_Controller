#include <Arduino.h>
#include <Servo.h>

#include "stdlib.h"						// memset(void * s int c, size_t n);

int
main( void )
{
	watchdogSetup();					// disable the watchdog
	init();								// initialize components, etc
	delay(1);							// give some time until doing other stuff
										//
	unsigned int angle = 90;			// define servo angle to spin at (speed); 90<right ; 90==stand_still; 90>left; 
	const unsigned int servoPin = 22;	// define arduino due GPIO pin to connect servo data line to
	Servo servo;						// define Servo object
	servo.attach(servoPin);				// assign servo pin to servo	
	servo.write(angle);					// set servo angle so its not moving
										//
	const int arraySize = 10;			// define size of array (size = 5 would be enough but eh)
	char array[arraySize];				// define buffer to receive serial data bytes
	char * array_ptr = array;			// define pointer to iterate over serial data array
	memset(array, '\0', arraySize);		// set initial array contents to null termination characters
										//
	int serialInput = 0;				// define variable to store serial data
	Serial.begin(9600);					// begin serial connection at 9600 baudrate
										//
	delay(1000); 						// grace period to let things start and fall into place (in milliseconds)

	for(;;)
	{
		if(Serial.available() > 0)									// if serial data is available ...
		{															//
			serialInput = Serial.read();							// read from serial
			if((serialInput == '\n') || (serialInput == '\r'))		// if serial data was newline or carriage return ...
			{
				if(array_ptr != array)				// if pointer is not equal to start of array ...
				{									//
					angle = atoi(array);			// convert array bytes to integer (=angle) (stitch together)
					servo.write(angle);				// write angle to servo
				}									//
				array_ptr = array;					// set pointer position to start of array
				memset(array, '\0', arraySize);		// reset contents of array back to null termination characters
			}
			else									// if data is not newline or carriage return ...
			{										//
				if(*array_ptr == '\0')				// if array address at pointer position is available ...
				{									//
					*array_ptr = serialInput;		// write serial data to pointer position
					array_ptr++;					// increment pointer
				}
			}
		}
	}
	
	servo.detach();
	
	return 0;
}
