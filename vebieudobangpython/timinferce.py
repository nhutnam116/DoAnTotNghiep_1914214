import cv2
import ultralytics
from ultralytics import YOLO

# Load the three YOLO models
model1 = YOLO(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11m\train\weights\best.pt")
model2 = YOLO(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11s\train\weights\best.pt")
model3 = YOLO(r"C:\Users\Admin\Downloads\DoanTN\train_21_11_2024\ketquatrain11n\train\weights\best.pt")

# Open the video using OpenCV
video_path = r"C:\Users\Admin\Downloads\video_2024-11-20_22-24-51.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()
cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
# Read the first frame
ret, frame = cap.read()
if ret:
    # Perform prediction on the first frame using all three models
    results1 = model1.predict(source=frame, conf=0.8, show=False)
    results2 = model2.predict(source=frame, conf=0.8, show=False)
    results3 = model3.predict(source=frame, conf=0.8, show=False)
    
    # Display the frame with predictions for model 1
    frame1 = results1[0].plot()  # Get the frame with predictions for model1
    cv2.imshow("Model M Prediction", frame1)

    # Display the frame with predictions for model 2
    frame2 = results2[0].plot()  # Get the frame with predictions for model2
    cv2.imshow("Model S Prediction", frame2)

    # Display the frame with predictions for model 3
    frame3 = results3[0].plot()  # Get the frame with predictions for model3
    cv2.imshow("Model N Prediction", frame3)

    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

else:
    print("Error: Unable to read the first frame.")
    cap.release()
    exit()

# Release the video capture object
cap.release()
