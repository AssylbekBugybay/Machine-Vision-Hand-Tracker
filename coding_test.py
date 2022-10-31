# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:59:39 2022

This exercise is based on the the YouTube tutorial of Nicholas Renotte 
@author: KSN2RT
"""
import cv2
import mediapipe as mp
import csv 
from datetime import datetime, date

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon=detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self. results = self.hands.process(imgRGB)      

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        #xmin, xmax = min(xList), max(xList)
        #ymin, ymax = min(yList), max(yList)
        #bbox = xmin, ymin, xmax, ymax
        
        self.lmList = lmList
        return lmList
    
cap = cv2.VideoCapture(0) # Maybe you have to use another ID as 0 for your camera depending on your system, try 1 or 2  
detector = handDetector()
  
# CSV-File Example Header 
csv_header = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
              "INDEX_FINGER_MCP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", 
              "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", 
             "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", 
             "PINKY_FINGER_MCP", "PINKY_FINGER_PIP", "PINKY_FINGER_DIP", "PINKY_FINGER_TIP"] 

while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img, True)
    lmList = detector.findPosition(img, 0, True)
    
    if len(lmList) != 0:
        print(lmList) 
    
    """
    Code something here 
    """

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Pressing q on keyboard interupts the while loop 
        break
    