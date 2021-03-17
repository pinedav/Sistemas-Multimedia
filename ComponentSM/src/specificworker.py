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
from PySide2.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D)
from PySide2.Qt3DCore import (Qt3DCore)
from PySide2.Qt3DExtras import (Qt3DExtras)
from genericworker import *
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

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

class Window(Qt3DExtras.Qt3DWindow):
    def __init__(self):
        super(Window, self).__init__()

        # Camera
        self.camera().lens().setPerspectiveProjection(45, 16 / 9, 0.1, 1000)
        self.camera().setPosition(QVector3D(0, 0, 40))
        self.camera().setViewCenter(QVector3D(0, 0, 0))

        # For camera controls
        self.createScene()
        #self.addSphere(2,-2,-2,2,10,10)
        self.camController = Qt3DExtras.QOrbitCameraController(self.rootEntity)
        self.camController.setLinearSpeed(50)
        self.camController.setLookSpeed(180)
        self.camController.setCamera(self.camera())

        self.setRootEntity(self.rootEntity)


    def createScene(self):
        # Root entity
        self.rootEntity = Qt3DCore.QEntity()

        # Material
        self.material = Qt3DExtras.QPhongMaterial(self.rootEntity)


        # Sphere
        #self.sphereEntity = Qt3DCore.QEntity(self.rootEntity)
        #self.sphereMesh = Qt3DExtras.QSphereMesh()
        #self.sphereMesh.setRadius(3)
        #self.sphereTransform = Qt3DCore.QTransform()

        #self.sphereEntity.addComponent(self.sphereMesh)
        #self.sphereEntity.addComponent(self.sphereTransform)
        #self.sphereEntity.addComponent(self.material)

    def add_cylinder(self, radius, length, x, y, z, rings=10, slices=10):
        self.cylinderEntity = Qt3DCore.QEntity(self.rootEntity)
        self.cylinderMesh = Qt3DExtras.QCylinderMesh()
        self.cylinderMesh.setRadius(radius)
        self.cylinderMesh.setLength(length)
        self.cylinderMesh.setSlices(rings)
        self.cylinderMesh.setRings(slices)
        self.cylinderTransform = Qt3DCore.QTransform()
        self.cylinderTransform.setTranslation(QVector3D(x, y, z))

        self.cylinderEntity.addComponent(self.cylinderMesh)
        self.cylinderEntity.addComponent(self.material)
        self.cylinderEntity.addComponent(self.cylinderTransform)


    def addSphere(self,radius, x, y, z, rings=10, slices=10):


        self.sphereEntity = Qt3DCore.QEntity(self.rootEntity)
        self.sphereMesh = Qt3DExtras.QSphereMesh()
        self.sphereMesh.setRadius(radius)
        self.sphereMesh.setSlices(rings)
        self.sphereMesh.setRings(slices)
        self.sphereTransform = Qt3DCore.QTransform()
        self.sphereTransform.setTranslation(QVector3D(x, y, z))

        self.sphereEntity.addComponent(self.sphereMesh)
        self.sphereEntity.addComponent(self.material)
        self.sphereEntity.addComponent(self.sphereTransform)


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


        self.view = Window()
        self.view.show()



        return True

    @QtCore.Slot()
    def compute(self):

        self.prueba()

        if self.new_image:
            self.count += 1

            self.cameras[self.imgCruda.cameraID] = np.frombuffer(self.imgCruda.image, np.uint8).reshape(
                self.imgCruda.height, self.imgCruda.width, self.imgCruda.depth)
            cv2.putText(self.cameras[self.imgCruda.cameraID], str(self.total) + ' fps', (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(self.cameras[self.imgCruda.cameraID], str(len(self.people.peoplelist)) + ' bodies', (500, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

            if self.new_people and self.people.cameraId == self.imgCruda.cameraID:
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

            self.new_image = False

        if time.time() - self.begin > 1:
            self.total = self.count
            self.count = 0
            self.begin = time.time()

        self.view.addSphere(3,0,0,0,50,50)
        self.view.add_cylinder(3, 20, 3, 0, 0, 200, 2)



    def prueba(self):
        object2cam = pt.transform_from(
            pr.active_matrix_from_intrinsic_euler_xyz(np.array([1.25, -0.028, -0.06])),
            np.array([381, 761, -3173]))

        object2cam2 = pt.transform_from(
            pr.active_matrix_from_intrinsic_euler_xyz(np.array([-2.04, -3.13, 0.09])),
            np.array([-192, 254, -3408]))

        tm = TransformManager()
        tm.add_transform("object", "camera", object2cam)
        tm.add_transform("object", "camera2", object2cam2)

        p = pt.transform(tm.get_transform("camera", "object"), np.array([0, 0, 0, 1]))

        ax = tm.plot_frames_in("object", s=500)
        ax.scatter(p[0], p[1], p[2])
        ax.set_xlim((-1000, 3000))
        ax.set_ylim((-3000, 3000))
        ax.set_zlim((0, 4000))
        plt.draw()



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

