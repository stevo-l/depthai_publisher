#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import RegionOfInterest

from cv_bridge import CvBridge, CvBridgeError
import depthai as dai

import numpy as np
import serial

from pathlib import Path
import sys
import cv2
import time
import argparse
import json
import blobconverter


class DepthaiCamera():
    res = [416, 416]
    fps = 20.0

    pub_topic_img = '/depthai_node/image/compressed'
    pub_topic_int = '/depthai_node/int'

    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.pub_image = rospy.Publisher(
            self.pub_topic_img, CompressedImage, queue_size=10)

        self.pub_int = rospy.Publisher(
            self.pub_topic_int, RegionOfInterest, queue_size=10)

        rospy.loginfo(
            "Publishing images to rostopic: {}".format(self.pub_topic_img))

        self.br = CvBridge()

        rospy.on_shutdown(lambda: self.shutdown())

    def rgb_camera(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(self.res[0], self.res[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self.fps)

        # Def xout / xin
        ctrl_in = self.pipeline.createXLinkIn()
        ctrl_in.setStreamName("cam_ctrl")
        ctrl_in.out.link(cam_rgb.inputControl)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("video")

        cam_rgb.preview.link(xout_rgb.input)

##########################################################################################

        configPath = Path('/home/uavteam007/blob/yolov5.json')

        with configPath.open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        if "input_size" in nnConfig:
            W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        classes = metadata.get("classes", {})
        coordinates = metadata.get("coordinates", {})
        anchors = metadata.get("anchors", {})
        anchorMasks = metadata.get("anchor_masks", {})
        iouThreshold = metadata.get("iou_threshold", {})
        confidenceThreshold = metadata.get("confidence_threshold", {})
        print(metadata)

        nnMappings = config.get("mappings", {})
        global labels
        labels = nnMappings.get("labels", {})
        print(labels)

        # get model path
        nnPath = '/home/uavteam007/blob/yolov5_dai.blob'
        # sync outputs
        syncNN = True

        detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)
        nnOut = self.pipeline.create(dai.node.XLinkOut)
        nnOut.setStreamName("nn")

        detectionNetwork.setConfidenceThreshold(confidenceThreshold)
        detectionNetwork.setNumClasses(classes)
        detectionNetwork.setCoordinateSize(coordinates)
        detectionNetwork.setAnchors(anchors)
        detectionNetwork.setAnchorMasks(anchorMasks)
        detectionNetwork.setIouThreshold(iouThreshold)
        detectionNetwork.setBlobPath(nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        # Linking [Not sure if this is meant to be here]
        cam_rgb.preview.link(detectionNetwork.input)
        detectionNetwork.passthrough.link(xout_rgb.input)
        detectionNetwork.out.link(nnOut.input)
####################################################################################
    # def frameNorm(frame, bbox):
    #     normVals = np.full(len(bbox), frame.shape[0])
    #     normVals[::2] = frame.shape[1]
    #     return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # def displayFrame(name, frame, detections):
    #     color = (255, 0, 0)
    #     for detection in detections:
    #         bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    #         cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    #         cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    #     # Show the frame
    #     cv2.imshow(name, frame)

    def run(self):
        self.rgb_camera()
        # self.nn_camera()

        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0
        color2 = (255, 255, 255)

        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def displayFrame(self, name, frame, detections):
            color = (255, 0, 0)
            target = 0
            # print()
            if len(detections) == 0:
                target = 0
            else:
                target = detections[0]

            for detection in detections:
                bbox = frameNorm(
                    frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labels[detection.label], (bbox[0] +
                            10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%",
                            (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), color, 2)

            # Show the frame
            self.publish_to_ros(frame, target)

        with dai.Device(self.pipeline) as device:
            video = device.getOutputQueue(
                name="video", maxSize=1, blocking=False)
            qDet = device.getOutputQueue(
                name="nn", maxSize=4, blocking=False)

            while True:
                inDet = qDet.get()
                frame = video.get().getCvFrame()
                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

                if inDet is not None:
                    detections = inDet.detections
                    counter += 1

                if frame is not None:
                    displayFrame(self, "rgb", frame, detections)
                # self.publish_to_ros(frame)

    def publish_to_ros(self, frame, detections):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        # target = 0
        int_out = RegionOfInterest()
        if hasattr(detections, 'xmin'):
            int_out.x_offset = int(detections.xmin)
            int_out.y_offset = int(detections.ymin)
            print(detections.xmin)
        else:
            int_out.x_offset = 0
            int_out.y_offset = 0
        int_out.height = 416
        int_out.width = 416
        int_out.do_rectify
        self.pub_image.publish(msg_out)
        self.pub_int.publish(int_out)


    def shutdown(self):
        cv2.destroyAllWindows()


def main():
    rospy.init_node('depthai_node')

    dai_cam = DepthaiCamera()

    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()
