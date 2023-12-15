# import os
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from keras.models import load_model

# model_path = os.getenv(MODEL_PATH)
model = load_model('Hand Gesture Recognition.h5')
hand_gestures = {0: "01_palm", 1: "02_l", 2: "03_fist", 3: "04_fist_moved", 4: "05_thumb", 5: "06_index",
                 6: "07_ok", 7: "08_palm_moved", 8: "09_c", 9: "10_down"}

# dpg.create_context()
# dpg.create_viewport()
# dpg.setup_dearpygui()


# def predict_hand_gesture(sender, app_data):
#     model_prediction = model.predict(image[:1])
#     if model_prediction is not None:
#         dpg.set_value("Result", f"This Is A: {hand_gestures[int(model_prediction)]}")
#     else:
#         dpg.set_value("Status", "Status: There Was An Error Predicting The Class Of The Image!!")
#
#
# def load_img(sender, app_data):
#     img_path = dpg.get_value("Image_Path")
#     if img_path:
#         try:
#             global image
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (64, 64))
#             img = img.flatten()
#             image = np.array([img])
#             dpg.set_value("Status", f"Status: Loaded The Image successfully.")
#         except Exception as e:
#             dpg.set_value("Status", f"Status: Error: {str(e)}")
#     else:
#         dpg.set_value("Status", "Status: Please provide a valid file path.")


# def capture_photo(sender, app_data):
    # Camera Initialization
cap = cv2.VideoCapture(0)  # 0 --> Default camera

# Background subtractor using KNN to capture the gesture witin the frame
bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

while True:
    # Frame-by-frame capture
    ret, frame = cap.read()

    # Convert frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color in HSV range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to capture the skin color
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

    # Applying background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Combining the skin mask with the background subtractor mask
    mask_combined = cv2.bitwise_and(mask_skin, mask_skin, mask=fg_mask)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

    # Apply combined mask to the original frame
    segmented_hand = cv2.bitwise_and(frame, frame, mask=mask_combined)

    # Convert the segmented frame to grayscale
    gray = cv2.cvtColor(segmented_hand, cv2.COLOR_BGR2GRAY)

    # Resize the segmented frame to match the model's input size
    resized_frame = cv2.resize(gray, (150, 150))

    # Reshape the frame to match the input shape of the model
    input_data_arr = np.array(resized_frame)
    input_data = input_data_arr.reshape((1, 150, 150, 1))  # Ensure single channel

    # Model predictions
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)

    print("Raw Prediction:", prediction)

    # Map to the gesture
    predicted_gesture = hand_gestures[predicted_label]

    # Display frame with the predicted gesture
    cv2.putText(frame, f"Predicted Gesture: {predicted_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)
    print("Predicted Probabilities:", prediction)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
# print("hi")


# with dpg.window(label="Price Prediction", width=400, height=200):
#     dpg.add_button(label="Open Camera", callback=capture_photo)
# #     dpg.add_text("Enter The Path Of The Image:")
# #     dpg.add_input_text(label="Image Path", tag="Image_Path", width=400)
# #     dpg.add_button(label="Load Image", callback=load_img)
# #     dpg.add_text(label="", tag="Status")
# #     dpg.add_button(label="Classify The Image", callback=predict_hand_gesture)
# #     dpg.add_text(label="", tag="Result")
#
# dpg.show_viewport()
# dpg.start_dearpygui()
# dpg.destroy_context()
