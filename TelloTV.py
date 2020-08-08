from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import model_from_json

from os import listdir
import argparse


#-----------------------

target_size = (152, 152)

#-----------------------

# standard argparse stuff
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-d', '--distance', type=int, default=3,
    help='use -d to change the distance of the drone. Range 0-6')
parser.add_argument('-sx', '--saftey_x', type=int, default=100,
    help='use -sx to change the saftey bound on the x axis . Range 0-480')
parser.add_argument('-sy', '--saftey_y', type=int, default=55,
    help='use -sy to change the saftey bound on the y axis . Range 0-360')
parser.add_argument('-os', '--override_speed', type=int, default=1,
    help='use -os to change override speed. Range 0-3')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

# Speed of the drone
S = 20
S2 = 5
UDOffset = 150

# this is just the bound box sizes that openCV spits out *shrug*
faceSizes = [1026, 684, 456, 304, 202, 136, 90]

# These are the values in which kicks in speed up mode, as of now, this hasn't been finalized or fine tuned so be careful
# Tested are 3, 4, 5
acc = [500,250,250,150,110,70,50]

# Frames per second of the pygame window display
FPS = 25
dimensions = (960, 720)

# 
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def detectFace(img_path, target_size=(152, 152)):
	
	img = cv2.imread(img_path)
	
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	if len(faces) > 0:
		x,y,w,h = faces[0]
		
		margin = 0
		x_margin = w * margin / 100
		y_margin = h * margin / 100
		
		if y - y_margin > 0 and y+h+y_margin < img.shape[1] and x-x_margin > 0 and x+w+x_margin < img.shape[0]:
			detected_face = img[int(y-y_margin):int(y+h+y_margin), int(x-x_margin):int(x+w+x_margin)]
		else:
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		
		detected_face = cv2.resize(detected_face, target_size)
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		#normalize in [0, 1]
		img_pixels /= 255 
		
		return img_pixels
	else:
		raise ValueError("Face could not be detected in ", img_path,". Please confirm that the picture is a face photo.")

#-------------------------


#DeepFace model
base_model = Sequential()
base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
base_model.add(Flatten(name='F0'))
base_model.add(Dense(4096, activation='relu', name='F7'))
base_model.add(Dropout(rate=0.5, name='D0'))
base_model.add(Dense(8631, activation='softmax', name='F8'))

base_model.load_weights("weights/VGGFace2_DeepFace_weights_val-0.9034.h5")

#Drop F8 and D0 layers. F7 is the representation layer.
model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)

#------------------------
def l2_normalize(x):
	return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
	euclidean_distance = source_representation - test_representation
	euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
	euclidean_distance = np.sqrt(euclidean_distance)
	return euclidean_distance

#------------------------	

#put your user pictures in this path as name_of_user.jpg
user_pictures = "database/"

users = dict()

for file in listdir(user_pictures):
	user, extension = file.split(".")
	img_path = 'database/%s.jpg' % (user)
	img = detectFace(img_path)
	
	representation = model.predict(img)[0]
	
	users[user] = representation
	
print("user representations retrieved successfully")


# If we are to save our sessions, we need to make sure the proper directories exist
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)

class FrontEnd(object):
    
    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()
        
        should_stop = False
        imgCount = 0
        OVERRIDE = False
        oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()
        
        # Safety Zone X
        szX = args.saftey_x

        # Safety Zone Y
        szY = args.saftey_y
        
        if args.debug:
            print("DEBUG MODE ENABLED!")

        while not should_stop:
            self.update()

            if frame_read.stopped:
                frame_read.stop()
                break

            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frameRet = frame_read.frame
            img=frameRet
            vid = self.tello.get_video_capture()

            if args.save_session:
                cv2.imwrite("{}/tellocap{}.jpg".format(ddir,imgCount),frameRet)
            
            frame = np.rot90(frame)
            imgCount+=1

            time.sleep(1 / FPS)

            # Listen for key presses
            k = cv2.waitKey(20)

            # Press 0 to set distance to 0
            if k == ord('0'):
                if not OVERRIDE:
                    print("Distance = 0")
                    tDistance = 0

            # Press 1 to set distance to 1
            if k == ord('1'):
                if OVERRIDE:
                    oSpeed = 1
                else:
                    print("Distance = 1")
                    tDistance = 1

            # Press 2 to set distance to 2
            if k == ord('2'):
                if OVERRIDE:
                    oSpeed = 2
                else:
                    print("Distance = 2")
                    tDistance = 2
                    
            # Press 3 to set distance to 3
            if k == ord('3'):
                if OVERRIDE:
                    oSpeed = 3
                else:
                    print("Distance = 3")
                    tDistance = 3
            
            # Press 4 to set distance to 4
            if k == ord('4'):
                if not OVERRIDE:
                    print("Distance = 4")
                    tDistance = 4
                    
            # Press 5 to set distance to 5
            if k == ord('5'):
                if not OVERRIDE:
                    print("Distance = 5")
                    tDistance = 5
                    
            # Press 6 to set distance to 6
            if k == ord('6'):
                if not OVERRIDE:
                    print("Distance = 6")
                    tDistance = 6

            # Press T to take off
            if k == ord('t'):
                if not args.debug:
                    print("Taking Off")
                    self.tello.takeoff()
                    self.tello.get_battery()
                self.send_rc_control = True

            # Press L to land
            if k == ord('l'):
                if not args.debug:
                    print("Landing")
                    self.tello.land()
                self.send_rc_control = False

            # Press Backspace for controls override
            if k == 8:
                if not OVERRIDE:
                    OVERRIDE = True
                    print("OVERRIDE ENABLED")
                else:
                    OVERRIDE = False
                    print("OVERRIDE DISABLED")

            if OVERRIDE:
                # S & W to fly forward & back
                if k == ord('w'):
                    self.for_back_velocity = int(S * oSpeed)
                elif k == ord('s'):
                    self.for_back_velocity = -int(S * oSpeed)
                else:
                    self.for_back_velocity = 0

                # a & d to pan left & right
                if k == ord('d'):
                    self.yaw_velocity = int(S * oSpeed)
                elif k == ord('a'):
                    self.yaw_velocity = -int(S * oSpeed)
                else:
                    self.yaw_velocity = 0

                # Q & E to fly up & down
                if k == ord('e'):
                    self.up_down_velocity = int(S * oSpeed)
                elif k == ord('q'):
                    self.up_down_velocity = -int(S * oSpeed)
                else:
                    self.up_down_velocity = 0

                # c & z to fly left & right
                if k == ord('c'):
                    self.left_right_velocity = int(S * oSpeed)
                elif k == ord('z'):
                    self.left_right_velocity = -int(S * oSpeed)
                else:
                    self.left_right_velocity = 0

            # Quit the software
            if k == 27:
                should_stop = True
                break

            gray  = cv2.cvtColor(frameRet, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)

            # Target size
            tSize = faceSizes[tDistance]

            # These are our center dimensions
            cWidth = int(dimensions[0]/2)
            cHeight = int(dimensions[1]/2)

            noFaces = len(faces) == 0

            # if we've given rc controls & get face coords returned
            # if self.send_rc_control and not OVERRIDE:
                
            if not OVERRIDE:
                for (x, y, w, h) in faces:
                    if w > 50:
                    
                        cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) #draw rectangle to main image

                        # 
                        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
                        roi_color = frameRet[y:y+h, x:x+w]

                        # setting Face Box properties
                        fbCol = (255, 0, 0) #BGR 0-255 
                        fbStroke = 2
                        
                        # end coords are the end of the bounding box x & y
                        end_cord_x = x + w
                        end_cord_y = y + h
                        end_size = w*2
                        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                        try:
                            detected_face = cv2.resize(detected_face, target_size) #resize to 152x152
                            img_pixels = image.img_to_array(detected_face)
                            img_pixels = np.expand_dims(img_pixels, axis = 0)
                            img_pixels /= 255
                            distances = []
                            captured_representation = model.predict(img_pixels)[0]


                            for i in users:
                                user_name = i
                                source_representation = users[i]
                                
                                distance = findEuclideanDistance(l2_normalize(captured_representation), l2_normalize(source_representation))
                                distances.append(distance)
                            
                            is_found = False; index = 0
                            for i in users:
                                user_name = i
                                if index == np.argmin(distances):
                                    print(distances[index] )
                                    if distances[index] <= 0.67:
                                        
                                        print("detected: ",user_name, "(",distances[index],")")
                                        user_name = user_name.replace("_", "")
                                        similarity = distances[index]
                                        
                                        is_found = True
                                        
                                        break
                                    
                                index = index + 1
                            print(is_found)
                            if is_found:
                                 # these are our target coordinates
                                targ_cord_x = int((end_cord_x + x)/2)
                                targ_cord_y = int((end_cord_y + y)/2) + UDOffset

                                # This calculates the vector from your face to the center of the screen
                                vTrue = np.array((cWidth,cHeight,tSize))
                                vTarget = np.array((targ_cord_x,targ_cord_y,end_size))
                                vDistance = vTrue-vTarget

                                # 
                                if not args.debug:
                                    # for turning
                                    if vDistance[0] < -szX:
                                        self.yaw_velocity = S
                                        # self.left_right_velocity = S2
                                    elif vDistance[0] > szX:
                                        self.yaw_velocity = -S
                                        # self.left_right_velocity = -S2
                                    else:
                                        self.yaw_velocity = 0
                                    
                                    # for up & down
                                    if vDistance[1] > szY:
                                        self.up_down_velocity = S
                                    elif vDistance[1] < -szY:
                                        self.up_down_velocity = -S
                                    else:
                                        self.up_down_velocity = 0

                                    F = 0
                                    if abs(vDistance[2]) > acc[tDistance]:
                                        F = S

                                    # for forward back
                                    if vDistance[2] > 0:
                                        self.for_back_velocity = S + F
                                    elif vDistance[2] < 0:
                                        self.for_back_velocity = -S - F
                                    else:
                                        self.for_back_velocity = 0

                                # Draw the face bounding box
                                label = user_name+" ("+"{0:.2f}".format(similarity)+")"					
                                cv2.rectangle(frameRet, (x, y), (end_cord_x, end_cord_y), fbCol, fbStroke)
                                cv2.putText(frameRet, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), fbStroke, cv2.LINE_AA)
                                # Draw the target as a circle
                                cv2.circle(frameRet, (targ_cord_x, targ_cord_y), 10, (0,255,0), 2)

                                # Draw the safety zone
                                # cv2.rectangle(frameRet, (targ_cord_x - szX, targ_cord_y - szY), (targ_cord_x + szX, targ_cord_y + szY), (0,255,0), fbStroke)

                                # Draw the estimated drone vector position in relation to face bounding box
                                cv2.putText(frameRet,str(vDistance),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                                

                        except Exception as e:
                            print(str(e))
                
                        
                       

                # for (x,y,w,h) in faces:
                    # if w > 130: #discard small detected faces
                    #     cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) #draw rectangle to main image
                        
                    #     detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                    #     detected_face = cv2.resize(detected_face, target_size) #resize to 152x152
                        
                    #     img_pixels = image.img_to_array(detected_face)
                    #     img_pixels = np.expand_dims(img_pixels, axis = 0)
                    #     img_pixels /= 255
                        
                    #     captured_representation = model.predict(img_pixels)[0]
                        
                    #     distances = []

                    #     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    #     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

                    #     center_x = int(width / 2)
                    #     center_y = int(height / 2)
                    #     z_area = 0
                        
                    #     for i in users:
                    #         user_name = i
                    #         source_representation = users[i]
                            
                    #         distance = findEuclideanDistance(l2_normalize(captured_representation), l2_normalize(source_representation))
                    #         distances.append(distance)
                        
                    #     is_found = False; index = 0
                    #     for i in users:
                    #         user_name = i
                    #         if index == np.argmin(distances):
                    #             print(distances[index] )
                    #             if distances[index] <= 0.62:
                                    
                    #                 print("detected: ",user_name, "(",distances[index],")")
                    #                 user_name = user_name.replace("_", "")
                    #                 similarity = distances[index]
                                    
                    #                 is_found = True
                                    
                    #                 break
                                
                    #         index = index + 1
                        
                    #     if is_found:
                    #         offset_x = 0
                    #         offset_y = 0
                    #         z_area = w * h
                            
                    #         face_center_x = x + int(h/2)
                    #         face_center_y = y + int(w/2)

                    #         offset_x = face_center_x - center_x
                    #         # Add 30 so that the drone covers as much of the subject as possible
                    #         offset_y = face_center_y - center_y - 30
                    #         adjust_tello_position(offset_x, offset_y, z_area)

                    #         display_img = cv2.imread("database/%s.jpg" % user_name)
                    #         pivot_img_size = 112
                    #         display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
                                            
                    #         try:
                    #             resolution_x = img.shape[1]; resolution_y = img.shape[0]
                                
                    #             label = user_name+" ("+"{0:.2f}".format(similarity)+")"
                                
                    #             if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                    #                 #top right
                    #                 img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img
                    #                 cv2.putText(img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)					
                                    
                    #                 #connect face and text
                    #                 cv2.line(img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                    #                 cv2.line(img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)
                    #             elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                    #                 #bottom left
                    #                 img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img
                    #                 cv2.putText(img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)
                                    
                    #                 #connect face and text
                    #                 cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                    #                 cv2.line(img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)
                                    
                    #             elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                    #                 #top left
                    #                 img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img
                    #                 cv2.putText(img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)
                                    
                    #                 #connect face and text
                    #                 cv2.line(img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                    #                 cv2.line(img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)
                                    
                    #             elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                    #                 #bottom righ
                    #                 img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img
                    #                 cv2.putText(img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)
                                    
                    #                 #connect face and text
                    #                 cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                    #                 cv2.line(img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
                                
                    #         except Exception as e:
                    #             print("exception occured: ", str(e))
                        

                # if there are no faces detected, don't do anything
                if noFaces:
                    self.yaw_velocity = 0
                    self.up_down_velocity = 0
                    self.for_back_velocity = 0
                    print("NO TARGET")
                
            # Draw the center of screen circle, this is what the drone tries to match with the target coords
            cv2.circle(frameRet, (cWidth, cHeight), 10, (0,0,255), 2)

            dCol = lerp(np.array((0,0,255)),np.array((255,255,255)),tDistance+1/7)

            if OVERRIDE:
                show = "OVERRIDE: {}".format(oSpeed)
                dCol = (255,255,255)
            else:
                show = "AI: {}".format(str(tDistance))

            # Draw the distance choosen
            cv2.putText(frameRet,show,(32,664),cv2.FONT_HERSHEY_SIMPLEX,1,dCol,2)

            # Display the resulting frame
            cv2.imshow(f'Tello Tracking...',frameRet)

        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()
        
        # Call it always before finishing. I deallocate resources.
        self.tello.end()


    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

def lerp(a,b,c):
    return a + c*(b-a)

def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
