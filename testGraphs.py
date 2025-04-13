from cmu_graphics import *
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time

def onAppStart(app):
    # MediaPipe setup
    app.mpDrawing = mp.solutions.drawing_utils
    app.mpHolistic = mp.solutions.holistic
    app.mpFaceMesh = mp.solutions.face_mesh
    
    # Initialize webcam
    app.cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe models
    app.holistic = app.mpHolistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    app.faceMesh = app.mpFaceMesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # State variables
    app.isCalibrated = False
    app.showVideo = True
    app.postureScore = 100
    app.postureStatus = 'Waiting for calibration...'
    
    # Posture metrics
    app.earToShoulderXOffset = 0
    app.headTiltAngle = 0
    app.shoulderYDifference = 0
    app.chinToShoulderMidDist = 0
    
    # Calibration variables
    app.calibrationStep = 0  # 0: waiting, 1: collecting data
    app.calibrationPrompt = "Sit up as straight as possible, look into the camera, and press 'c' to start calibration"
    app.calibrationFrameCount = 0
    app.calibrationFramesNeeded = 30
    app.calibrationValues = []
    app.idealPostureMetrics = {
        'earToShoulderXOffset': 0,
        'headTiltAngle': 0,
        'shoulderYDifference': 0,
        'chinToShoulderMidDist': 0
    }
    app.postureWarnings = []
    
    # UI variables
    app.scoreHistory = []
    app.maxHistoryPoints = 50
    app.hasImage = False  # Flag to track if we have a valid image
    
    # Process first frame
    processFrame(app)
    app.steps = 0
    app.postureHistory = []

def processFrame(app):
    success, image = app.cap.read()
    if not success:
        app.hasImage = False
        return
    
    # Flip image horizontally
    image = cv2.flip(image, 1)
    
    # Convert to RGB
    imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    holisticResults = app.holistic.process(imageRgb)
    faceResults = app.faceMesh.process(imageRgb)
    
    # Process posture detection if landmarks detected
    if holisticResults.pose_landmarks and faceResults.multi_face_landmarks:
        # Get landmarks
        faceLandmarks = faceResults.multi_face_landmarks[0].landmark
        poseLandmarks = holisticResults.pose_landmarks.landmark
        
        # Get key points
        leftEar = faceLandmarks[234]  # Left ear
        rightEar = faceLandmarks[454]  # Right ear
        chin = faceLandmarks[152]  # Chin point
        nose = faceLandmarks[4]  # Nose tip
        
        leftShoulder = poseLandmarks[app.mpHolistic.PoseLandmark.LEFT_SHOULDER]
        rightShoulder = poseLandmarks[app.mpHolistic.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate ear midpoint and shoulder midpoint
        earMidpoint = ((leftEar.x + rightEar.x) / 2, (leftEar.y + rightEar.y) / 2)
        shoulderMidpoint = ((leftShoulder.x + rightShoulder.x) / 2, (leftShoulder.y + rightShoulder.y) / 2)
        
        # Calculate posture metrics
        currentMetrics = calculatePostureMetrics(
            leftEar, rightEar, chin, nose, 
            leftShoulder, rightShoulder,
            earMidpoint, shoulderMidpoint
        )
        
        # Update app metrics
        app.earToShoulderXOffset = currentMetrics['earToShoulderXOffset']
        app.headTiltAngle = currentMetrics['headTiltAngle']
        app.shoulderYDifference = currentMetrics['shoulderYDifference']
        app.chinToShoulderMidDist = currentMetrics['chinToShoulderMidDist']
        
        # Handle calibration or evaluation
        if app.isCalibrated:
            evaluatePosture(app, currentMetrics)
        else:
            handleCalibration(app, currentMetrics)
    
    # Store the image only when video should be shown
    if app.showVideo:
        app.image = image
        app.hasImage = True

def calculatePostureMetrics(leftEar, rightEar, chin, nose, leftShoulder, rightShoulder, earMidpoint, shoulderMidpoint):
    # 1. Horizontal offset between ear midpoint and shoulder midpoint (forward head)
    earToShoulderXOffset = earMidpoint[0] - shoulderMidpoint[0]
    
    # 2. Calculate head tilt angle (from vertical)
    x_diff = earMidpoint[0] - shoulderMidpoint[0]
    y_diff = earMidpoint[1] - shoulderMidpoint[1]
    headTiltAngle = np.degrees(np.arctan2(x_diff, -y_diff))
    
    # 3. Shoulder level (y-difference between shoulders)
    shoulderYDifference = abs(leftShoulder.y - rightShoulder.y)
    
    # 4. Distance between chin and shoulder midpoint (slouching)
    chinToShoulderMidDist = np.sqrt(
        ((chin.x - shoulderMidpoint[0]) ** 2) + 
        ((chin.y - shoulderMidpoint[1]) ** 2)
    )
    
    return {
        'earToShoulderXOffset': earToShoulderXOffset,
        'headTiltAngle': headTiltAngle,
        'shoulderYDifference': shoulderYDifference,
        'chinToShoulderMidDist': chinToShoulderMidDist
    }

def handleCalibration(app, currentMetrics):
    if app.calibrationStep == 1:
        # Collecting measurements
        app.calibrationFrameCount += 1
        app.calibrationValues.append(currentMetrics)
        
        if app.calibrationFrameCount >= app.calibrationFramesNeeded:
            # Calculate average values
            avgMetrics = {}
            for key in currentMetrics.keys():
                values = [m[key] for m in app.calibrationValues]
                avgMetrics[key] = sum(values) / len(values)
            
            # Set ideal metrics
            app.idealPostureMetrics = avgMetrics
            app.isCalibrated = True
            app.showVideo = False  # Hide video after calibration
            app.postureStatus = "Calibration complete!"

def warningMessages(app):
    app.warningMessages = []
    if app.headTilted:
        app.warningMessages.append('Straighten your head')
    if app.shouldersTilted:
        app.warningMessages.append('Straighten your shoulders')
    if app.slouching:
        app.warningMessages.appoend('Straighten your back')

def printWarningMessages(app):
    for message in app.warningMessages:
        app.width


def evaluatePosture(app, currentMetrics):
    app.headTilted = False
    app.shouldersTilted = False
    app.slouching = False
    # Forward head position score
    # idealOffset = app.idealPostureMetrics['earToShoulderXOffset']
    # offsetDiff = abs(currentMetrics['earToShoulderXOffset'] - idealOffset)
    # offsetScore = max(0, 100 - (offsetDiff * 300))
    # if offsetScore < 90:
    #     app.headTilted = True
    
    # Head tilt angle score
    idealAngle = app.idealPostureMetrics['headTiltAngle']
    angleDiff = abs(currentMetrics['headTiltAngle'] - idealAngle)
    angleScore = max(0, 100 - (angleDiff * 2))
    if angleScore < 90:
        app.headTilted = True
    
    # Shoulder level score
    idealShoulderDiff = app.idealPostureMetrics['shoulderYDifference']
    shoulderDiffError = abs(currentMetrics['shoulderYDifference'] - idealShoulderDiff)
    shoulderScore = max(0, 100 - (shoulderDiffError * 500))
    if shoulderScore < 90:
        app.shouldersTilted = True
    
    # Chin to shoulder distance score (slouching)
    idealChinDist = app.idealPostureMetrics['chinToShoulderMidDist']
    chinDistRatio = currentMetrics['chinToShoulderMidDist'] / idealChinDist if idealChinDist > 0 else 1
    if app.slouching < 90:
        app.slouching = True
    
    if chinDistRatio < 1:  # Slouching
        chinDistScore = max(0, 100 - (100 * (1 - chinDistRatio) * 2))
    else:  # Extended or normal
        chinDistScore = max(0, 100 - (100 * (chinDistRatio - 1)))
    
    # Weighted combined score using the same parameters
    app.postureScore = int(
        #offsetScore * 0.3 +      # Forward head
        angleScore * 0.3 +       # Head tilt
        shoulderScore * 0.3 +    # Shoulder level
        chinDistScore * 0.4      # Slouching
    )

    app.postureHistory.append((time.time(), app.postureScore))


    
    # Add to history for graph
    app.scoreHistory.append(app.postureScore)
    if len(app.scoreHistory) > app.maxHistoryPoints:
        app.scoreHistory.pop(0)
    
    # Determine posture status
    if app.postureScore > 90:
        app.postureStatus = "Excellent Posture"
    elif app.postureScore > 80:
        app.postureStatus = "Good Posture"
    elif app.postureScore > 70:
        app.postureStatus = "Fair Posture"
    elif app.postureScore > 60:
        app.postureStatus = "Poor Posture"
    else:
        app.postureStatus = "Very Poor Posture"

def onStep(app):
    app.steps += 1
    if app.steps % 10 == 0:  # Process every 5 frames for better performance
        processFrame(app)

def onKeyPress(app, key):
    if key == 'c':  # Start calibration
        app.calibrationStep = 1
        app.calibrationFrameCount = 0
        app.calibrationValues = []
        app.calibrationPrompt = "Hold your ideal posture for calibration..."
        app.showVideo = True  # Show video during calibration
    
    elif key == 'v':  # Toggle video visibility
        app.showVideo = not app.showVideo
    
    elif key == 'r':  # Reset calibration
        app.isCalibrated = False
        app.calibrationStep = 0
        app.calibrationPrompt = "Sit up straight and press 'c' to start calibration"
        app.showVideo = True
    
    elif key == 'q':  # Quit
        app.cap.release()
        app.holistic.close()
        app.faceMesh.close()
        app.stop()

def drawGraph(app):
    # Graph dimensions
    graphLeft = 50
    graphTop = 300
    graphWidth = app.width - 100
    graphHeight = 100

    drawRect(graphLeft, graphTop, graphWidth, graphHeight, fill='lightgrey')
    drawLine(graphLeft, graphTop + graphHeight, graphLeft + graphWidth, graphTop + graphHeight)  # x-axis
    drawLine(graphLeft, graphTop, graphLeft, graphTop + graphHeight)  # y-axis

    if len(app.postureHistory) < 2:
        drawLabel("Not enough data", graphLeft + graphWidth / 2, graphTop + graphHeight / 2)
        return

    # Extract data
    times = [t for (t, s) in app.postureHistory]
    scores = [s for (t, s) in app.postureHistory]
    minTime = min(times)
    maxTime = max(times)
    timeRange = max(maxTime - minTime, 1)
    scoreRange = 100  # postureScore is 0–100
    
    # Draw the line
    

    for y in range(20, 101, 20):
        lineY = graphTop + graphHeight - (y / 100 * graphHeight)
        drawLine(graphLeft+2, lineY, graphLeft + graphWidth, lineY, 
                fill=rgbToColor(220, 220, 220), dashes=False)
        drawLabel(str(y), graphLeft - 20, lineY, size=10)
   
    # Scale points for plotting
    scaledPoints = []
    for (time, score) in app.postureHistory:
        x = graphLeft + ((time - minTime) / timeRange) * graphWidth
        y = graphTop + graphHeight - ((score / scoreRange) * graphHeight)
        scaledPoints.append((x, y))

    for i in range(1, len(scaledPoints)):
        x0, y0 = scaledPoints[i - 1]
        x1, y1 = scaledPoints[i]
        drawLine(x0, y0, x1, y1, fill='blue', lineWidth=2)

 

    # Show current score
    latestTime, latestScore = app.postureHistory[-1]
    drawLabel(f"Current: {int(latestScore)}", graphLeft + graphWidth - 60, graphTop + 10, size=12, fill='black')






def drawPostureScore(app):
    scoreColor = rgbToColor(46, 204, 113) if app.postureScore > 80 else \
                rgbToColor(230, 126, 34) if app.postureScore > 60 else \
                rgbToColor(231, 76, 60)
    
    # Background panel
    drawRect(0, 0, app.width, app.height, fill=rgbToColor(240, 240, 240))
    
    # Header
    drawRect(0, 0, app.width, 60, fill=rgbToColor(52, 73, 94))
    drawLabel("Posture Monitor", app.width/2, 30, fill='white', size=24, bold=True)
    
    # Current score display
    drawCircle(app.width/2, 150, 80, fill='white', border=scoreColor, borderWidth=8)
    drawLabel(str(app.postureScore), app.width/2, 150, size=40, bold=True, fill=scoreColor)
    drawLabel(app.postureStatus, app.width/2, 250, size=20, bold=True, fill=scoreColor)
    
    # # Score history graph
    # if len(app.scoreHistory) > 1:
    #     graphWidth = app.width - 100
    #     graphHeight = 100
    #     graphX = 50
    #     graphY = 300
        
    #     # Graph background
    #     drawRect(graphX, graphY, graphWidth, graphHeight, fill='white', border='lightGray')
        
    #     # Draw graph lines
    #     for y in range(0, 101, 20):
    #         lineY = graphY + graphHeight - (y / 100 * graphHeight)
    #         drawLine(graphX, lineY, graphX + graphWidth, lineY, 
    #                 fill=rgbToColor(200, 200, 200), dashes=True)
    #         drawLabel(str(y), graphX - 20, lineY, size=10)
        
    #     # Plot score history
    #     if len(app.scoreHistory) > 0:
    #         pointSpacing = graphWidth / max(1, min(app.maxHistoryPoints, len(app.scoreHistory) - 1))
    #         for i in range(len(app.scoreHistory) - 1):
    #             x1 = graphX + i * pointSpacing
    #             y1 = graphY + graphHeight - (app.scoreHistory[i] / 100 * graphHeight)
    #             x2 = graphX + (i + 1) * pointSpacing
    #             y2 = graphY + graphHeight - (app.scoreHistory[i + 1] / 100 * graphHeight)
    #             drawLine(x1, y1, x2, y2, fill=scoreColor, lineWidth=3)


    drawGraph(app)
    
    # Metrics display
    metricsY = 450
    metricBoxWidth = app.width / 2 - 40
    
    # Forward head and head tilt
    drawRect(20, metricsY, metricBoxWidth, 80, fill='white', border='lightGray')
    drawLabel("Head Position", 20 + metricBoxWidth/2, metricsY + 20, bold=True)
    drawLabel(f"Forward: {abs(app.earToShoulderXOffset):.2f}", 
             20 + metricBoxWidth/2, metricsY + 45, size=12)
    drawLabel(f"Tilt: {abs(app.headTiltAngle):.2f}°", 
             20 + metricBoxWidth/2, metricsY + 65, size=12)
    
    # Shoulder level and slouching
    drawRect(app.width - 20 - metricBoxWidth, metricsY, metricBoxWidth, 80, fill='white', border='lightGray')
    drawLabel("Body Position", app.width - 20 - metricBoxWidth/2, metricsY + 20, bold=True)
    drawLabel(f"Shoulders: {abs(app.shoulderYDifference):.2f}", 
             app.width - 20 - metricBoxWidth/2, metricsY + 45, size=12)
    drawLabel(f"Slouching: {abs(app.chinToShoulderMidDist):.2f}", 
             app.width - 20 - metricBoxWidth/2, metricsY + 65, size=12)
    
    # Controls
    controlsY = app.height - 40
    drawLabel("Press 'v' to toggle video | 'r' to recalibrate | 'q' to quit", 
             app.width/2, controlsY, size=14)

def redrawAll(app):
    if app.showVideo and app.hasImage:
        # Display video feed (during calibration)
        imageRGB = cv2.cvtColor(app.image, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(imageRGB)
        cmuImage = CMUImage(pilImage)
        
        # Scale image to fit
        imgHeight, imgWidth = app.image.shape[:2]
        scale = min(app.width / imgWidth, app.height / imgWidth)
        newWidth = int(imgWidth * scale)
        newHeight = int(imgHeight * scale)
        
        # Draw the image centered
        drawImage(cmuImage, (app.width - newWidth) / 2, (app.height - newHeight) / 2, 
                 width=newWidth, height=newHeight)
        
        # Calibration overlay
        if not app.isCalibrated:
            drawRect(0, app.height - 60, app.width, 60, fill='black', opacity=70)
            drawLabel(app.calibrationPrompt, app.width/2, app.height - 30, 
                     bold=True, fill='white', size=18)
            
            if app.calibrationStep == 1 and app.calibrationFrameCount > 0:
                # Progress bar - ensuring width is always positive
                progressWidth = max(1, (app.calibrationFrameCount / app.calibrationFramesNeeded) * (app.width - 100))
                drawRect(50, app.height - 80, app.width - 100, 10, fill=None, border='white')
                drawRect(50, app.height - 80, progressWidth, 10, fill='white')
    else:
        drawPostureScore(app)

def rgbToColor(r, g, b):
    return rgb(r, g, b)

def main():
    runApp(width=640, height=560)

if __name__ == "__main__":
    main()
