from imutils.video import VideoStream
from collections import deque
from datetime import datetime
import numpy as np
import argparse
import imutils
import serial
import time
import cv2

# https://www.youtube.com/watch?v=HORX7UQAHgM

#############################################################################
# function declaration														#
#############################################################################

def draw_circle(event,x,y,flags,param):
	global mouseX;
	global mouseY;
	mouseX,mouseY = x,y

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

#############################################################################
# variable declaration														#
#############################################################################

frameName = "Harold";							# window name
cv2.namedWindow(frameName);						# create window
cv2.setMouseCallback(frameName,draw_circle);	# link mouse to window

mouseX = 0;						# new setpoint posX
mouseY = 0;						# new setpoint posY (unused)

maxBuffer = 64;						# max number of points to draw / iterate over
lowerBlue = (90, 100, 100);			# lower HSV color space boundary (blue)
upperBlue = (110, 255, 255);		# upper HSV color space boundary (blue)

lowerGreen = (50, 100, 100);		# lower HSV color space boundary (green)
upperGreen = (70, 255, 255);		# upper HSV color space boundary (green)


ptsBlue = deque(maxlen = maxBuffer)		# points to draw / iterate over
ptsGreen = deque(maxlen = maxBuffer)	# points to draw / iterate over

vs = VideoStream(src=0).start()								# start webcam stream	
s = serial.Serial(port="COM3", baudrate=9600, bytesize=8) 	# start serial connection (port, baudrate, bytesize)

time.sleep(1.0)	# give time for the camera to start

#############################################################################
# PID stuff																	#
#############################################################################

kp = 2.3;			# proportion
ki = 0.02;			# integral
kd = 5.3;			# derivative

error = 0.0;		# distance between set_point and positionX
prev_error = 0.0;	# previous error
i_error = 0.0;		# error over time

pid = 0;			# pid formula output
dt = 0.01;			# delta_time

positions = []		# list of positions for filtering outliers

frame_size = 640;				# window width (proportional height)
set_point = frame_size / 2;		# initial set_point is middle of window

beamPosY = 0;		# positionY of support beam
beamMid = 328;		# positionY of support beam when level

isStart = False;	# start sending servo instructions

#############################################################################
# Main loop																	#
#############################################################################

while True:
	frame = vs.read()											# get frame from webcam stream
	if frame is not None:										# if there is a frame
		frame = imutils.resize(frame, width=frame_size)			# resize frame
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)			# blur frame
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)			# convert to HSV color space
		maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue )		# construct mask for ball blob
		maskGreen = cv2.inRange(hsv, lowerGreen, upperGreen )	# construct mask for beam blob
		contoursBlue = cv2.findContours(maskBlue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 	# create ball contours
		contoursBlue = imutils.grab_contours(contoursBlue)												# find ball contours
		contoursGreen = cv2.findContours(maskGreen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 	# create beam contours
		contoursGreen = imutils.grab_contours(contoursGreen)											# find beam contours
		centerBlue = None										# initialize center value to None
		centerGreen = None										# initialize center value to None
		if len(contoursGreen) > 0:								# if there are contours for beam blob
			c = max(contoursGreen, key=cv2.contourArea)		# get largest contour in mask
			((x, y), radius) = cv2.minEnclosingCircle(c)	# get x and y position and center of contour
			M = cv2.moments(c)	# get image moments	(Cx = M10/M00; Cy = M01/M00)
			m00 = M["m00"];		# M00
			m01 = M["m01"];		# M01
			m10 = M["m10"];		# M10			
			if m00: 														# make sure m00 > 0 to avoid null-division
				centerGreen = (int(m10 / m00), int(m01 / m00));					# calculate center position
				beamPosY = beamMid - centerGreen[1];							# set relative beam posY to mid point
				cv2.circle(frame, (int(x), int(y)), int(1), (255, 0, 0), 10)	# draw circle around blob
		if len(contoursBlue) > 0:												# if there are contours for blue ball blob
			c = max(contoursBlue, key=cv2.contourArea)						# get largest contour in mask
			((x, y), radius) = cv2.minEnclosingCircle(c)					# get x and y position and center of contour
			M = cv2.moments(c)	# get image moments	(Cx = M10/M00; Cy = M01/M00)
			m00 = M["m00"];		# M00
			m01 = M["m01"];		# M01
			m10 = M["m10"];		# M10			
			if m00: 										# make sure m00 > 0 to avoid null-division
				centerBlue = (int(m10 / m00), int(m01 / m00));	# calculate center position			
				positions.append(centerBlue[0]);				# append x-position to list of positions			
				if(isStart):									# if user pressed 's' key to start pid calculations
					if(len(positions) >= 3):					# if positions has 3 or more positions ...
						selection_sort(positions);				# sort list of positions
						error = (positions[1] - set_point);		# calculate error
						positions.clear();						# clear list of positions so it can fill up again
						velocity = int(error - prev_error);		# get ball 'velocity'
						prev_error = error;						# set previous_error to error
						error = (error / (frame_size/2)) * 90	# throttle error between 0-90-ish
						i_error += error						# sum of errors over time
						i_error *= 0.1							# extinction
						
						if(abs(error) < 2) : error = 0;			# clamp error to prevent unneccesary oscilation

						angle = int(90 + (((error-beamPosY/2) * kp) + (i_error * ki) + ((velocity-beamPosY/2) * kd)));
						
						if(angle < 0) : angle = 0;								# angle cannot be lower than 0
						if(angle > 180) : angle = 180;							# angle cannot be higher than 180
						if((beamPosY < -40) and (angle < 90)) : angle = 90;		# prevent servo from ripping apart structure
						if((beamPosY > 40) and (angle > 90)) : angle = 90;		# prevent servo from ripping apart structure

						print("error:" + str(error) + \
							  " out: " + str(angle) + \
							  " beam:" + str(beamPosY));
						s.write(str(angle).encode());					# send angle value over serial
						s.write(b'\r\n');								# signal end of transmission with '\r\n' closing bytes
#						if(s.inWaiting() > 0):							# wait for incoming serial data
#							data = s.readline();						# read serial data
#							data = str(data).strip('b\'\\r\\n');		# strip serial data to leave just the numbers
#							if(len(output) <= 3):						# filter leftover bullshit
#								data = int(data);						# convert output to integer
				else:								# if no angle is being transmitted over serial ...
					pid = 90;						# set angle to stationary
					s.write(str(pid).encode());		# transmit angle to make sure servo is no longer running
					s.write(b'\r\n');				# signal end of transmission with '\r\n' closing bytes
					
			if radius > 5:															# if ball blob radius is > 5 ...
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)	# draw circle around ball blob
		ptsBlue.appendleft(centerBlue) 												# update points queue
		for i in range(1, len(ptsBlue)):											# loop over tracked points
			if ptsBlue[i - 1] is None or ptsBlue[i] is None: continue;				# ignore points with type None
			thickness = int(np.sqrt(maxBuffer / float(i + 1)) * 2.5)				# calculate line thickness
			cv2.line(frame, ptsBlue[i - 1], ptsBlue[i], (0, 0, 255), thickness)		# draw trailing red line in center
		cv2.imshow(frameName, frame) 												# draw frame
		key = cv2.waitKey(1) & 0xFF															# get keyboard input
		if key == ord("q"):	break; 															# q == quit
		if key == ord("s"):	isStart = True; 												# s == start PID
		if key == ord("d"):	isStart = False; 												# s == start PID
		elif key == ord('a'): set_point = mouseX; print ("new set point: " + str(mouseX));	# a == get new set point

#############################################################################
# cleanup																	#
#############################################################################

pid = 90;						# set servo angle to stationary
s.write(str(pid).encode());		# stop servo
s.write(b'\r\n');				# signal end of serial transmission

vs.stop()					# stop webcam stream
cv2.destroyAllWindows()		# close opencv window