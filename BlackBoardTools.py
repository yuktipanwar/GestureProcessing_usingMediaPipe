# http://192.168.252.97:8080/video

import threading
import time
import datetime
import math
import numpy as np
import cv2
import mediapipe as mp
import imutils
import os
import tkinter as tk
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageFont, ImageDraw
import emoji
import subprocess
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# imports end   here


uri_cam_feed = "http://100.68.40.220:8080/video"
#uri_cam_feed = 0
pointerOn = not True
clickOn = True
painterOn = not  True
eraserOn = not True
draw_painting = not True
drawMenu = True
click_status = False

# This code is for the server 
# Lets import the libraries
import socket, cv2, pickle,struct,imutils

# Socket Create

# server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# host_name  = socket.gethostname()
# host_ip = socket.gethostbyname(host_name)
# print('HOST IP:',host_ip)
# port = 9999
# socket_address = (host_ip,port)

# # Socket Bind
# server_socket.bind(socket_address)

# # Socket Listen
# server_socket.listen(5)
# print("LISTENING AT:",socket_address)


class FPS:

    # This class reads FPS
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            # self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# Classes END....

# Funcitons start here.....
def callback():
    startTracking()



def distanceCalc(a, b):
    distance = (a[0]*a[0] - b[0]*b[0]) + (a[1]*a[1] - b[1]*b[1])
    if distance < 0:
        distance = distance*(-1)
    if distance == 0:
        return 0
    distance = math.sqrt(distance)
    return distance


def angleCalc(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angleoutput = np.abs(radians*180.0/np.pi)
    if angleoutput > 180.0:
        angleoutput = 360.0 - angleoutput
    # print("Angle = "+str(angleoutput))
    return angleoutput

option_square_size = (100,100)

def draw_menu(frame, pointer_option = False, pen_option = False, color_selction_option= False, eraser_option= False):
    # Drawing the whole menu
    raw_d = (frame.shape[0],frame.shape[1])
    
    global emoji_font_click
    
    #           height          width
    menu_start = (raw_d[1]-option_square_size[0],0)
    menu_stop = (raw_d[1],raw_d[0])
    cv2.rectangle(frame, menu_start, menu_stop, (0,255,0), 2)
    if pointer_option:
        cv2.rectangle(frame, (raw_d[1]-option_square_size[0],0),(raw_d[1],option_square_size[0]), (255,0,0),5)
    if pen_option:
        cv2.rectangle(frame, (raw_d[1]-option_square_size[0],option_square_size[0]),(raw_d[1],option_square_size[0]*2), (255,255,0),5)
    if color_selction_option:
        cv2.rectangle(frame, (raw_d[1]-option_square_size[0],option_square_size[0]*2),(raw_d[1],option_square_size[0]*3), (0,255,0),5)
    if eraser_option:
        cv2.rectangle(frame, (raw_d[1]-option_square_size[0],option_square_size[0]*3),(raw_d[1],option_square_size[0]*4), (255,0,255),5)
    return frame

def menu_trigger_and_activator(frame,track):
    # print(f"CLick @ x : {track[0]}, Y : {track[1]}")
    raw_d = (frame.shape[0],frame.shape[1])
    global pointerOn
    global clickOn
    global painterOn
    global eraserOn
    global draw_painting
    global drawMenu
    global current_color
    if track[0]>raw_d[1]-option_square_size[0]:
        cv2.putText(frame,"In Menu",(130,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        
    # return frame
        
        pointerOn = False
        clickOn = True
        painterOn = False
        eraserOn = False
        draw_painting = True
        # drawMenu = False

        if option_square_size[0]*0<track[1]<option_square_size[0]*1:
            clickOn = True    
        if option_square_size[0]*1<track[1]<option_square_size[0]*2:
            painterOn =True
            draw_painting = True
            current_color = 'g'
            # clickOn = False
        if option_square_size[0]*2<track[1]<option_square_size[0]*3:
            painterOn =True
            draw_painting = True
            current_color = 'r'

        if option_square_size[0]*3<track[1]<option_square_size[0]*4:
            eraserOn = True
            # clickOn =  False
        
    return frame

def drawPointer(result,frame, sendFrame=False):
    # indexFingerLocation 
    for handLms in result.multi_hand_landmarks:
        raw_dimensions = (frame.shape[0], frame.shape[1])
        indexfingerloc = (int(handLms.landmark[8].x*raw_dimensions[1]),int(handLms.landmark[8].y*raw_dimensions[0]))
        if sendFrame:
            cv2.circle(frame, indexfingerloc, 10, (0,0,255),2)

        # print(raw_dimensions)
    return indexfingerloc,frame

def drawClick(result, frame, sendFrame=False):
    # indexFingerLocation 
    global click_status
    for handLms in result.multi_hand_landmarks:
        raw_dimensions = (frame.shape[0], frame.shape[1])
        A = (int(handLms.landmark[8].x*raw_dimensions[1]),int(handLms.landmark[8].y*raw_dimensions[0]))
        B = (int(handLms.landmark[12].x*raw_dimensions[1]),int(handLms.landmark[12].y*raw_dimensions[0]))
        distance = int(distanceCalc(A,B))
        midPoint = (int((A[0]+B[0])/2),int((A[1]+B[1])/2))
        
        if 0<distance<120:
            clickStatus="Click : ON"
            click_status = True
            frame =menu_trigger_and_activator(frame,midPoint)
        else:
            click_status = False
            clickStatus="Click : OFF"

        if sendFrame:
            cv2.circle(frame, A, 6, (0,0,255),2)
            cv2.circle(frame, B, 6, (0,0,255),2)
            cv2.circle(frame, midPoint, 4, (0,255,0),2)
            cv2.putText(frame,f"{distance}",midPoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(frame,clickStatus,(10,20), cv2.FONT_HERSHEY_SIMPLEX,.5 , (0,0,0), 1, cv2.LINE_AA)
            cv2.line(frame, A, B, (0,0,0), 2) 

        # print(raw_dimensions)
    return midPoint,frame

pointsPainted = []
def paintViaPoniter(result,frame,color):
    track,_ = drawPointer(result, frame, False)
    if track not in pointsPainted and click_status:
        pointsPainted.append((track[0],track[1],color))
    # Draw the points painted:
def eraseViaEraser(result,frame):
    eraser_size = (100,100)
    point,_ = drawPointer(result, frame, False)
    eraser_start = (int(point[0] - eraser_size[0]/2),int(point[1] - eraser_size[1]/2))
    eraser_end = (int(point[0] + eraser_size[0]/2),int(point[1] + eraser_size[1]/2))
    cv2.rectangle(frame, eraser_start, eraser_end, (255,255,255),-1)
    global pointsPainted
    for pt in pointsPainted:
        if eraser_start[0]<pt[0]<eraser_end[0] and eraser_start[1]<pt[1]<eraser_end[1]:
            pointsPainted.remove(pt)
            # Splicing mode
            # try:
            #     pointsPainted = pointsPainted[:pointsPainted.index(pt)]
            # except:
            #     pass
            pass

    return frame
_ct = 0

emoji_font_click = "✌🏻"
emoji_font_pointer = "👆🏻"
emoji_font_pencil = "✏️"
emoji_font_eraser = "⬜️"

def startTracking():
    print('\n[i]\t\tTracking intiated.')
    try:
        vs = WebcamVideoStream(src=uri_cam_feed).start()
    except:
        vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    global results_pose    
    global pointerOn
    global clickOn
    global painterOn
    global eraserOn
    global draw_painting
    global drawMenu
    global KillMeOnce
    global current_color

    KillMeOnce = "KillED"
    hand = mp_hands.Hands(max_num_hands=1)
    god_loop_run = True
    
    track_hands = True
    current_color = 'g'
    # client_socket,addr = server_socket.accept()
    while god_loop_run:
        image = vs.read()
        image = cv2.flip(image, 1)

        image = imutils.resize(image, height=400)
        # source_image = cv2.imread('src_img.png', cv2.IMREAD_COLOR)
        # print(len(source_image[2]))
        # source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        # raw_dimensions = (image.shape[0], image.shape[1])
        print(f"Mainloop : {KillMeOnce}")
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if drawMenu:
            draw_menu(image, True, True, True,True)
        if track_hands:
            results_hands = hand.process(image)
            if results_hands.multi_hand_landmarks:
                if pointerOn:
                    index_location,image = drawPointer(results_hands,image,True)
                if clickOn:
                    midPointLoc, image = drawClick(results_hands, image, True)
                if painterOn:
                    paintViaPoniter(results_hands,image,current_color)
                if eraserOn:
                    image = eraseViaEraser(results_hands, image)
                for handLms in results_hands.multi_hand_landmarks:
                    #mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
                    pass
        if draw_painting:
            l = len(pointsPainted)
            for i in range(l-1):
                if pointsPainted[i][2] == 'r':
                    cv2.circle(image,(pointsPainted[i][0],pointsPainted[i][1]), 6, (255,0,0), -1)
                if pointsPainted[i][2] == 'g':
                    cv2.circle(image,(pointsPainted[i][0],pointsPainted[i][1]), 6, (0,255,0), -1)
                if pointsPainted[i][2] == 'b':
                    cv2.circle(image,(pointsPainted[i][0],pointsPainted[i][1]), 6, (0,0,255), -1)
                
                paint_point_distance = distanceCalc((pointsPainted[i][0],pointsPainted[i][1]), (pointsPainted[i+1][0],pointsPainted[i+1][1]))
                
                if paint_point_distance<150:
                    if pointsPainted[i][2] == 'r':
                        cv2.line(image, (pointsPainted[i][0],pointsPainted[i][1]), (pointsPainted[i+1][0],pointsPainted[i+1][1]), (255,0,0), 6)
                    if pointsPainted[i][2] == 'g':
                        cv2.line(image, (pointsPainted[i][0],pointsPainted[i][1]), (pointsPainted[i+1][0],pointsPainted[i+1][1]), (0,255,0), 6)
                    if pointsPainted[i][2] == 'b':
                        cv2.line(image, (pointsPainted[i][0],pointsPainted[i][1]), (pointsPainted[i+1][0],pointsPainted[i+1][1]), (0,0,255), 6)
        ## Use simsum.ttc to write Chinese.
        # Drawing the icons
        font = ImageFont.truetype("./static/NotoEmoji-Bold.ttf",50)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((640,25),emoji_font_click,(0,0,0),font=font)
        draw.text((640,125),emoji_font_pencil,(0,0,0),font=font)
        draw.text((640,225),emoji_font_pointer,(0,0,0),font=font)
        draw.text((640,325),emoji_font_eraser,(255,255,255),font=font)
        image = np.array(img_pil)
        # ENDS
        # dest = cv2.addWeighted(source_image, 0.5, image, 0.5, 0.0)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # send_frame = image
        # a = pickle.dumps(send_frame)
        # message = struct.pack("Q",len(a))+a
        # client_socket.sendall(message)
		
        cv2.imshow('MediaPipe Pose', image)
        # cv2.imshow('With Image', dest)
        # cv2.imshow('MediaPipe Pose', image)
        # print(f'{image.shape[0]} ,{image.shape[1]}')
        if cv2.waitKey(10) & 0xFF == 27:
            # client_socket.close()
            break

        fps.update()
        # print()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


# Funcitons end here....
callback()
