#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *
import cv2
import numpy as np

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
sys.path.append('/opt/robocomp/lib')

COCO_IDS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"]
SKELETON_CONNECTIONS = [("left_ankle", "left_knee"),
                        ("left_knee", "left_hip"),
                        ("right_ankle", "right_knee"),
                        ("right_knee", "right_hip"),
                        ("left_hip", "right_hip"),
                        ("left_shoulder", "left_hip"),
                        ("right_shoulder", "right_hip"),
                        ("left_shoulder", "right_shoulder"),
                        ("left_shoulder", "left_elbow"),
                        ("right_shoulder", "right_elbow"),
                        ("left_elbow", "left_wrist"),
                        ("right_elbow", "right_wrist"),
                        ("left_eye", "right_eye"),
                        ("nose", "left_eye"),
                        ("nose", "right_eye"),
                        ("left_eye", "left_ear"),
                        ("right_eye", "right_ear"),
                        ("left_ear", "left_shoulder"),
                        ("right_ear", "right_shoulder")]


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.new_image = False
        self.contPersonas = 0
        self.xMano = 0
        self.yMano = 0
        self.Period = 50
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        self.params = params
        self.viewimage = "true" in self.params["viewimage"]
        return True


    @QtCore.Slot()
    def compute(self):
        #print('SpecificWorker.compute...')
        #self.Almac_Personas()
        #print(self.contPersonas)
        #self.Gesto1()
        #for Num in range(self.contPersonas):
            #print(self.peopleAux.peoplelist[Num].joints['right_wrist'].y)
            #print(self.peopleAux.peoplelist[Num].joints['left_wrist'].y)

        if self.new_image:
          self.imgCV = np.frombuffer(self.imgCruda.image, np.uint8).reshape(self.imgCruda.height, self.imgCruda.width, self.imgCruda.depth)
          self.new_image = False
        if self.new_people:
            self.ProcesImg(self.people, self.imgCV)
            cv2.imshow("Camera " + str(self.imgCruda.cameraID), self.imgCV)
            cv2.waitKey(1)
        return True

    def Almac_Personas(self):
        self.contPersonas = len(self.peopleAux.peoplelist)
        self.contPersonas = self.contPersonas(se)

    def Gesto1(self):

        for Num in range(self.contPersonas):
            #print(self.peopleAux.peoplelist[Num].joints['right_wrist'].y)
            if self.peopleAux.peoplelist[Num].joints['right_wrist'].y > 1000:
                print("Hola")
            #print(self.peopleAux.peoplelist[Num].joints['left_wrist'].y)
            if self.peopleAux.peoplelist[Num].joints['left_wrist'].y > 1000:
                print("Adios")


    def ProcesImg(self, people, image):
        # draw
        if self.viewimage:
            for person in people.peoplelist:
                for name1, name2 in SKELETON_CONNECTIONS:
                    try:
                        joint1 = person.joints[name1]
                        joint2 = person.joints[name2]
                        if joint1.score > 0.5:
                            cv2.circle(image, (joint1.i, joint1.j), 10, (0, 0, 255))
                        if joint2.score > 0.5:
                            cv2.circle(image, (joint2.i, joint2.j), 10, (0, 0, 255))
                        if joint1.score > 0.5 and joint2.score > 0.5:
                            cv2.line(image, (joint1.i, joint1.j), (joint2.i, joint2.j), (0, 255, 0), 2)
                    except:
                        pass
    #################################################################################################
    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #

    # SUBSCRIPTION to pushRGBD method from CameraRGBDSimplePub interface
    #
    def CameraRGBDSimplePub_pushRGBD(self, im, dep):

        #print("publish")
        self.imgCruda=im
        self.new_image=True

    def HumanCameraBody_newPeopleData(self, people):
        self.people = people
        self.new_people=True


    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompHumanCameraBody you can use this types:
    # RoboCompHumanCameraBody.TImage
    # RoboCompHumanCameraBody.TGroundTruth
    # RoboCompHumanCameraBody.KeyPoint
    # RoboCompHumanCameraBody.Person
    # RoboCompHumanCameraBody.PeopleData

