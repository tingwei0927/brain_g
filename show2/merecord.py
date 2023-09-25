import cv2
import time
import sys
start_time = time.time()
cap = cv2.VideoCapture(0)
recording = False
fps = cap.get(cv2.CAP_PROP_FPS)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH,5000)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,5000)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = "E://things/master/DIH/DIH/show2/testvideo.mp4"
out = cv2.VideoWriter(video, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
a123 = ""
framecount = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    
    ret, frame = cap.read()
    elapsed_time1 = time.time() - start_time
    
    if not ret:
        print("Cannot receive frame")
        #print(size)
        break
    if a123 == "": #用來擋，卡在第一幀不動，有輸入才會繼續錄
        a123 = input()
    framecount = framecount+1
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
    print("print",frame.shape)
    out.write(frame)       # 將取得的每一幀圖像寫入空的影片
    cv2.imshow('gogogo', frame)
    if cv2.waitKey(5) == ord('q'):
        break
    if framecount/fps >10:
        break             # 按下 q 鍵停止

cap.release()
out.release()      # 釋放資源
cv2.destroyAllWindows()
elapsed_time2 = time.time()-start_time
print("time:",framecount/fps)
print("222")
print("ddd")
print("framecount",framecount)
print("good")
print("ok")
