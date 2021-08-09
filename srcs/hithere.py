#!/usr/bin/env python3

import cv2
import depthai as dai
import math
import numpy as np
from checkerboard import detect_checkerboard


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()

edgeDetectorRgb = pipeline.createEdgeDetector()

xoutEdgeRgb = pipeline.createXLinkOut()
xoutVideo = pipeline.createXLinkOut()
xinEdgeCfg = pipeline.createXLinkIn()

edgeRgbStr = "edge rgb"
edgeCfgStr = "edge cfg"
videoStr = "Video"


xoutEdgeRgb.setStreamName(edgeRgbStr)
xinEdgeCfg.setStreamName(edgeCfgStr)
xoutVideo.setStreamName(videoStr)

# xoutVideo.input.setBlocking(False)
# xoutVideo.input.setQueueSize(1)

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)


edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())
# print(camRgb.getVideoWidth(), camRgb.getVideoHeight())

# Linking
camRgb.video.link(edgeDetectorRgb.inputImage)
camRgb.video.link(xoutVideo.input)

edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)

xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

	# Output/input queues
	edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 8, False)
	edgeCfgQueue = device.getInputQueue(edgeCfgStr)
	videoQueue =  device.getOutputQueue(videoStr, maxSize=1, blocking=False)

	print("Switch between sobel filter kernels using keys '1' and '2'")

	while(True):
		edgeRgb = edgeRgbQueue.get()
		edgeRgbFrame = edgeRgb.getFrame()
		filterEdgeFrame = np.copy(edgeRgbFrame)
		filterEdgeFrame[filterEdgeFrame < 150] = 0
		# edgeRgbFrame[edgeRgbFrame < 150] = 0
		videoFrame = videoQueue.get().getCvFrame()
		cv2.imshow(edgeRgbStr, edgeRgbFrame)
		# print(f"vid shape: {videoFrame.shape}. EdgeShape: {edgeRgbFrame.shape}")
		# print(type(edgeRgbFrame))
		
	

		# Show the frame

		# LineFrame = cv2.cvtColor(edgeRgbFrame, cv2.COLOR_GRAY2BGR)
		# LineFrame = np.copy(edgeRgbFrame)
		lines = cv2.HoughLinesP(filterEdgeFrame, 1, np.pi/180, threshold = 100, minLineLength=250, maxLineGap=50)
		if lines is not None:
			print(len(lines))
			for i in range(0, len(lines)):
				l = lines[i][0]
				cv2.line(videoFrame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
		cv2.imshow("HoughlinesP edge", videoFrame)

		# ret, corners = cv2.findChessboardCorners(videoFrame, (7,7))
		# print(f"ret: {ret}, corners:\n{corners}")
		# cv2.drawChessboardCorners(videoFrame, (7,7), corners, ret)

		# if (ret):
		# 	cv2.imshow("ChessBoardCorners", videoFrame)

		# lines = cv2.HoughLines(edgeRgbFrame, 1, np.pi / 180, 200)
    
		# if lines is not None:
		# 	print(len(lines))
		# 	for i in range(0, len(lines)):
		# 		rho = lines[i][0][0]
		# 		theta = lines[i][0][1]
		# 		a = math.cos(theta)
		# 		b = math.sin(theta)
		# 		x0 = a * rho
		# 		y0 = b * rho
		# 		pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
		# 		pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
		# 		cv2.line(videoFrame, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

		# cv2.imshow("Corners", videoFrame)
		# cv2.cvtColor(videoFrame, cv2.COLOR_RGB2GRAY)
		# print("MAMARASEASE\n")

		key = cv2.waitKey(1)
		if key == ord('q'):
			break

		if key == ord('1'):
			print("Switching sobel filter kernel.")
			cfg = dai.EdgeDetectorConfig()
			sobelHorizontalKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
			sobelVerticalKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
			cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
			edgeCfgQueue.send(cfg)

		if key == ord('2'):
			print("Switching sobel filter kernel.")
			cfg = dai.EdgeDetectorConfig()
			sobelHorizontalKernel = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
			sobelVerticalKernel = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
			cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
			edgeCfgQueue.send(cfg)