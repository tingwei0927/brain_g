import cv2

cap = cv2.VideoCapture(0) # 使用默认设备索引0，如果有多个相机设备，可能需要调整索引值
# cap.set(cv2.CAP_PROP_FPS, 15)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot receive frame")
        break
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()