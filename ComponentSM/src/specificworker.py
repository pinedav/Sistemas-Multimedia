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
import time

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
        self.count = 0
        self.total = 0
        self.begin = time.time()
        self.cameras = {}

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

    # @QtCore.Slot()
    # def compute(self):
    #     #print('SpecificWorker.compute...')
    #     #self.Almac_Personas()
    #     #print(self.contPersonas)
    #     #self.Gesto1()
    #     #for Num in range(self.contPersonas):
    #         #print(self.peopleAux.peoplelist[Num].joints['right_wrist'].y)
    #         #print(self.peopleAux.peoplelist[Num].joints['left_wrist'].y)
    #
    #     if self.new_image:
    #       self.imgCV = np.frombuffer(self.imgCruda.image, np.uint8).reshape(self.imgCruda.height, self.imgCruda.width, self.imgCruda.depth)
    #       self.new_image = False
    #     if self.new_people:
    #         self.ProcesImg(self.people, self.imgCV)
    #         cv2.imshow("Camera " + str(self.imgCruda.cameraID), self.imgCV)
    #         cv2.waitKey(1)
    #     return True

    @QtCore.Slot()
    def compute(self):
        self.count += 1

        if self.new_image:
            self.cameras[self.imgCruda.cameraID] = np.frombuffer(self.imgCruda.image, np.uint8).reshape(
                self.imgCruda.height, self.imgCruda.width, self.imgCruda.depth)
            cv2.putText(self.cameras[self.imgCruda.cameraID], str(self.total) + ' fps', (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(self.cameras[self.imgCruda.cameraID], str(len(self.people.peoplelist)) + ' bodies', (500, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            self.new_image = False

            if self.new_people:
                #self.transform_to_world(self.people)
                self.cameras[self.imgCruda.cameraID] = self.draw_body(self.people, self.cameras[self.imgCruda.cameraID])

            self.cameras[self.imgCruda.cameraID] = cv2.cvtColor(self.cameras[self.imgCruda.cameraID], cv2.COLOR_BGR2RGB)
            pix = QPixmap.fromImage(
                QImage(self.cameras[self.imgCruda.cameraID], self.cameras[self.imgCruda.cameraID].shape[1],
                       self.cameras[self.imgCruda.cameraID].shape[0], QImage.Format_RGB888))
            if self.imgCruda.cameraID == 1:
                self.ui.label_img_left.setPixmap(pix)
            if self.imgCruda.cameraID == 3:
                self.ui.label_img_right.setPixmap(pix)

        if time.time() - self.begin > 1:
            self.total = self.count
            self.count = 0
            self.begin = time.time()

    def transform_to_world(self, people):
        for person in people.peoplelist:
            for name1, name2 in SKELETON_CONNECTIONS:
                try:
                    joint1 = person.joints[name1]
                    joint2 = person.joints[name2]
                    if joint1.score > 0.5 and joint2.score > 0.5:
                        p = pt.transform(self.tm.get_transform("camera_1", "object"),
                                         np.array([joint1.x, joint1.y, joint1.z, 1]))

                        ax = self.tm.plot_frames_in("object", s=500)
                        ax.scatter(p[0], p[1], p[2])
                        ax.set_xlim((-1000, 3000))
                        ax.set_ylim((-3000, 3000))
                        ax.set_zlim((-2000.0, 4000))
                        plt.show()
                except:
                    pass

        # Draw body parts on image

    def draw_body(self, people, image):
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
        return image

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

