# # import cv2

# # def main():
# #     cap = cv2.VideoCapture(0)

# #     if not cap.isOpened():
# #         print("ERROR: Cannot open webcam")
# #         return

# #     ret, frame = cap.read()
# #     if not ret:
# #         print("ERROR: Cannot read from webcam")
# #         cap.release()
# #         return

# #     h, w = frame.shape[:2]

# #     # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     # out = cv2.VideoWriter('webcam_output.mp4', fourcc, 30.0, (w, h))
# #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
# #     out = cv2.VideoWriter('webcam_output.avi', fourcc, 30.0, (w, h))

# #     print("Recording... Press ESC to stop")

# #     try:
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             out.write(frame)

# #             cv2.imshow("Webcam", frame)

# #             if cv2.waitKey(1) & 0xFF == 27:  # ESC
# #                 break

# #     finally:
# #         cap.release()
# #         out.release()
# #         cv2.destroyAllWindows()
# #         print("Saved to webcam_output.mp4")


# # if __name__ == "__main__":
# #     main()


# import cv2

# # Initialize webcam (0 is usually the default camera)
# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# # 'XVID' is a common codec for .avi files
# # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))


# print("Recording... Press 'q' to stop.")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Write the frame to the file
#     out.write(frame)

#     # Display the resulting frame
#     cv2.imshow('Webcam Recording', frame)

#     # Press 'q' on the keyboard to stop recording
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release everything when done
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# import cv2

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# print("Recording... Press 'q' to stop.")

# # Use VP8 codec for WebM
# fourcc = cv2.VideoWriter_fourcc(*'VP80')

# out = cv2.VideoWriter('output.webm', fourcc, 20.0, (640, 480))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     out.write(frame)

#     cv2.imshow('Webcam Recording', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()








import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Recording... Press 'q' to stop.")

# Use VP8 codec for WebM
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.imshow('Webcam Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()