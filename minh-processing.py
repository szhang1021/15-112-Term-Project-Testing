from cmu_graphics import *
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from feedback import getPositiveFeedback 
import time

def onAppStart(app):
    # Initialize MediaPipe components
    app.mpDrawing = mp.solutions.drawing_utils
    app.mpDrawingStyles = mp.solutions.drawing_styles
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
    
    # Initialize posture variables
    app.postureScore = 100
    app.postureStatus = "Waiting for calibration..."
    
    # Posture metrics
    app.earToShoulderXOffset = 0
    app.headTiltAngle = 0
    app.shoulderYDifference = 0
    app.chinToShoulderMidDist = 0
    
    # Calibration variables
    app.isCalibrated = False
    app.calibrationStep = 0
    app.calibrationPrompts = [
        "Sit up as straight as possible",
        "Look forward with your chin parallel to the floor",
        "Keep your shoulders level",
        "Hold this posture - Calibrating..."
    ]
    app.idealPostureMetrics = {
        'earToShoulderXOffset': 0,
        'headTiltAngle': 0,
        'shoulderYDifference': 0,
        'chinToShoulderMidDist': 0
    }
    app.calibrationFrameCount = 0
    app.calibrationFramesNeeded = 30
    app.calibrationValues = []
    
    # Initial image processing
    processFrame(app)
    app.steps = 0


    # Display Encouragement Messaging
    app.displayMessage = False
    app.message = ' '
    app.postureType = ' '

    app.postureHistory = []

def processFrame(app):
    success, image = app.cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        return
    
    # Flip the image horizontally for mirror-like display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Holistic
    holisticResults = app.holistic.process(imageRgb)
    
    # Process with MediaPipe Face Mesh
    faceResults = app.faceMesh.process(imageRgb)
    
    # Draw face landmarks
    if faceResults.multi_face_landmarks:
        for faceLandmarks in faceResults.multi_face_landmarks:
            app.mpDrawing.draw_landmarks(
                image=image,
                landmark_list=faceLandmarks,
                connections=app.mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=app.mpDrawingStyles.get_default_face_mesh_tesselation_style()
            )
    
    # Process posture detection
    if holisticResults.pose_landmarks and faceResults.multi_face_landmarks:
        # Draw pose landmarks
        app.mpDrawing.draw_landmarks(
            image,
            holisticResults.pose_landmarks,
            app.mpHolistic.POSE_CONNECTIONS,
            landmark_drawing_spec=app.mpDrawingStyles.get_default_pose_landmarks_style()
        )
        
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
        
        # Convert landmarks to image coordinates
        h, w, c = image.shape
        
        # Calculate ear midpoint (for head position)
        earMidpoint = (
            (leftEar.x + rightEar.x) / 2,
            (leftEar.y + rightEar.y) / 2
        )
        
        # Convert to pixel coordinates
        chinPoint = (int(chin.x * w), int(chin.y * h))
        nosePoint = (int(nose.x * w), int(nose.y * h))
        leftEarPoint = (int(leftEar.x * w), int(leftEar.y * h))
        rightEarPoint = (int(rightEar.x * w), int(rightEar.y * h))
        earMidPoint = (int(earMidpoint[0] * w), int(earMidpoint[1] * h))
        leftShoulderPoint = (int(leftShoulder.x * w), int(leftShoulder.y * h))
        rightShoulderPoint = (int(rightShoulder.x * w), int(rightShoulder.y * h))
        
        # Calculate shoulder midpoint
        shoulderMidpoint = (
            (leftShoulder.x + rightShoulder.x) / 2,
            (leftShoulder.y + rightShoulder.y) / 2
        )
        shoulderMidPoint = (int(shoulderMidpoint[0] * w), int(shoulderMidpoint[1] * h))
        
        # Draw lines for visualization
        cv2.line(image, leftEarPoint, leftShoulderPoint, (0, 255, 255), 2)
        cv2.line(image, rightEarPoint, rightShoulderPoint, (0, 255, 255), 2)
        cv2.line(image, leftShoulderPoint, rightShoulderPoint, (0, 255, 255), 2)
        cv2.line(image, chinPoint, shoulderMidPoint, (0, 255, 0), 2)
        cv2.line(image, earMidPoint, shoulderMidPoint, (255, 0, 0), 2)
        
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
        if not app.isCalibrated:
            handleCalibration(app, currentMetrics)
        else:
            evaluatePosture(app, currentMetrics)
        
        # Add visual indicators to the image
        # Horizontal line for shoulder level
        cv2.line(image, (0, leftShoulderPoint[1]), (w, leftShoulderPoint[1]), (0, 0, 255), 1)
        cv2.line(image, (0, rightShoulderPoint[1]), (w, rightShoulderPoint[1]), (0, 0, 255), 1)
        
        # Add metrics to image
        metricsY = 30
        cv2.putText(image, f"Score: {app.postureScore}", (w - 150, metricsY), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw calibration prompt if not calibrated
        if not app.isCalibrated:
            promptY = h - 40
            cv2.putText(image, app.calibrationPrompts[app.calibrationStep], (10, promptY),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show progress bar for current calibration step
            if app.calibrationStep == 3:  # Only in the final step
                progressWidth = int((app.calibrationFrameCount / app.calibrationFramesNeeded) * (w - 20))
                cv2.rectangle(image, (10, promptY + 10), (10 + progressWidth, promptY + 20), (0, 255, 0), -1)
    
    # Store the image directly
    app.image = image

def calculatePostureMetrics(leftEar, rightEar, chin, nose, leftShoulder, rightShoulder, earMidpoint, shoulderMidpoint):
    # 1. Horizontal offset between ear midpoint and shoulder midpoint (forward head)
    earToShoulderXOffset = earMidpoint[0] - shoulderMidpoint[0]
    
    # 2. Calculate head tilt angle (from vertical)
    # Vector from shoulder midpoint to ear midpoint
    x_diff = earMidpoint[0] - shoulderMidpoint[0]
    y_diff = earMidpoint[1] - shoulderMidpoint[1]
    headTiltAngle = np.degrees(np.arctan2(x_diff, -y_diff))  # Negative y because image coordinates
    
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
    # Handle each calibration step
    if app.calibrationStep < 3:
        # Wait for key press to move to next step
        return
    elif app.calibrationStep == 3:
        # Collecting calibration measurements
        app.calibrationFrameCount += 1
        app.calibrationValues.append(currentMetrics)
        
        if app.calibrationFrameCount >= app.calibrationFramesNeeded:
            # Calculate average values from collected frames
            avgMetrics = {}
            for key in currentMetrics.keys():
                values = [m[key] for m in app.calibrationValues]
                avgMetrics[key] = sum(values) / len(values)
            
            # Set ideal metrics
            app.idealPostureMetrics = avgMetrics
            app.isCalibrated = True
            app.postureStatus = "Calibration complete!"
            print("Posture calibration complete!")

def evaluatePosture(app, currentMetrics):
    badPostureTypes = ""

    # 1. Forward head position
    idealOffset = app.idealPostureMetrics['earToShoulderXOffset']
    offsetDiff = abs(currentMetrics['earToShoulderXOffset'] - idealOffset)
    offsetScore = max(0, 100 - (offsetDiff * 300))
    if offsetScore < 70:
        badPostureTypes = "Forward Head Posture"

    # 2. Head tilt
    idealAngle = app.idealPostureMetrics['headTiltAngle']
    angleDiff = abs(currentMetrics['headTiltAngle'] - idealAngle)
    angleScore = max(0, 100 - (angleDiff * 2))
    if angleScore < 80:
        badPostureTypes = "Head Tilt"

    # 3. Shoulder level
    idealShoulderDiff = app.idealPostureMetrics['shoulderYDifference']
    shoulderDiffError = abs(currentMetrics['shoulderYDifference'] - idealShoulderDiff)
    shoulderScore = max(0, 100 - (shoulderDiffError * 500))
    if shoulderScore < 70:
        badPostureTypes = "Uneven Shoulders"

    # 4. Chin to shoulder distance
    idealChinDist = app.idealPostureMetrics['chinToShoulderMidDist']
    chinDistRatio = currentMetrics['chinToShoulderMidDist'] / idealChinDist if idealChinDist > 0 else 1
    if chinDistRatio < 1:
        chinDistScore = max(0, 100 - (100 * (1 - chinDistRatio) * 2))
    else:
        chinDistScore = max(0, 100 - (100 * (chinDistRatio - 1)))
    if chinDistScore < 75:
        badPostureTypes = "Slouched Posture"

    # Overall score
    app.postureScore = int(
        offsetScore * 0.3 +
        angleScore * 0.2 +
        shoulderScore * 0.2 +
        chinDistScore * 0.3
    )

    app.postureHistory.append((time.time(), app.postureScore))

    # Save the issues detected directly in the app object
    app.postureTypes = badPostureTypes
    
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
        app.postureStatus = "Bad Posture - Fix Now!"

def onStep(app):
    app.steps += 1
    if app.steps % 8 == 0:
        processFrame(app)

def onKeyPress(app, key):
    if key == 'c':  # Press 'c' to start/restart calibration
        app.isCalibrated = False
        app.calibrationStep = 0
        app.calibrationFrameCount = 0
        app.calibrationValues = []
        print("Starting posture calibration...")
    
    if key == 'n' and not app.isCalibrated:  # 'n' to move to next calibration step
        app.calibrationStep = min(3, app.calibrationStep + 1)
        app.calibrationFrameCount = 0
        print(f"Calibration step: {app.calibrationStep + 1}/4")
    
    if key == 'q':  # Press 'q' to quit
        app.cap.release()
        app.holistic.close()
        app.faceMesh.close()
        app.stop()

    if key == 'f': # Press 'f' to display encouragement message
        app.displayMessage = not app.displayMessage
        app.message = getPositiveFeedback(app.postureTypes)

def drawGraph(app):
    # Graph dimensions
    graphLeft = 50
    graphTop = 300
    graphWidth = 300
    graphHeight = 150

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

    # Scale points for plotting
    scaledPoints = []
    for (time, score) in app.postureHistory:
        x = graphLeft + ((time - minTime) / timeRange) * graphWidth
        y = graphTop + graphHeight - ((score / scoreRange) * graphHeight)
        scaledPoints.append((x, y))

    # Draw the line
    for i in range(1, len(scaledPoints)):
        x0, y0 = scaledPoints[i - 1]
        x1, y1 = scaledPoints[i]
        drawLine(x0, y0, x1, y1, fill='blue', lineWidth=2)

    # Draw circles every 5 seconds on the time axis (assuming 30 fps = 150 steps)
    interval = 5
    nextTick = ((minTime // interval) + 1) * interval  # first tick after minTime

    while nextTick < maxTime:
        x = graphLeft + ((nextTick - minTime) / timeRange) * graphWidth
        y = graphTop + graphHeight
        drawCircle(x, y, 4, fill='red')
        nextTick += interval

    # Show current score
    latestTime, latestScore = app.postureHistory[-1]
    drawLabel(f"Current: {int(latestScore)}", graphLeft + graphWidth - 60, graphTop + 10, size=12, fill='black')


def redrawAll(app):
    if hasattr(app, 'image'):
        # Convert OpenCV image to PIL image, then to CMUImage
        imageRGB = cv2.cvtColor(app.image, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(imageRGB)
        cmuImage = CMUImage(pilImage)
        
        # Draw camera feed
        drawImage(cmuImage, 0, 0)
        
        # Draw posture information overlay
        drawRect(10, 10, 250, 100, fill='white', opacity=70)
        
        # Draw calibration status or posture score
        if not app.isCalibrated:
            drawLabel("CALIBRATION MODE", 130, 25, size=14, bold=True, fill='red')
            drawLabel(f"Step {app.calibrationStep + 1}/4", 130, 45, size=12)
            drawLabel(app.calibrationPrompts[app.calibrationStep], 130, 65, size=10)
            drawLabel("Press 'n' for next step", 130, 85, size=10)
        else:
            # Posture score color
            scoreColor = 'green'
            if app.postureScore < 80:
                scoreColor = 'orange'
            if app.postureScore < 60:
                scoreColor = 'red'
            
            drawLabel(f"Posture Score: {app.postureScore}", 130, 25, size=14, bold=True, fill=scoreColor)
            drawLabel(app.postureStatus, 130, 45, size=12, fill=scoreColor)
            
            # Draw metrics
            metricsText = f"Head Forward: {abs(app.earToShoulderXOffset):.2f}"
            drawLabel(metricsText, 130, 65, size=10)
            
            metricsText = f"Shoulder Level: {abs(app.shoulderYDifference):.2f}"
            drawLabel(metricsText, 130, 85, size=10)
            
            # Draw a small posture score bar
            drawRect(10, 110, 250, 10, fill=None, border='black')
            barWidth = min(250, max(0, app.postureScore * 2.5))
            drawRect(10, 110, barWidth, 10, fill=scoreColor)
            
            # Display encouraging message
            if app.displayMessage:
                drawLabel(app.message, app.width/2, 65, size = 16)

            drawGraph(app)

        # Instructions at bottom
        drawLabel("Press 'c' to calibrate, 'q' to quit", app.width/2, app.height - 20, 
                 size=12, bold=True)


        

    else:
        drawLabel("Starting camera...", app.width/2, app.height/2, size=20)

def main():
    runApp(width=640, height=480)

if __name__ == "__main__":
    main()