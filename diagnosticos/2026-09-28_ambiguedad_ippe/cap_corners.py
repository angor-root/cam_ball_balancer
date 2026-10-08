import cv2, numpy as np, time, json
CAM='/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0918_20201224_v008-video-index0'
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
p=cv2.aruco.DetectorParameters_create(); p.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
cap=cv2.VideoCapture(CAM,cv2.CAP_V4L2); cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')); cap.set(3,640); cap.set(4,480)
out=[]; t0=time.time(); saved=0
while time.time()-t0<15:
    ok,f=cap.read()
    if not ok: continue
    c,ids,_=cv2.aruco.detectMarkers(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),d,parameters=p)
    if ids is None or len(ids)<4: continue
    out.append({str(int(i)):cc.reshape(4,2).tolist() for i,cc in zip(ids.ravel(),c)})
    if saved<1 and time.time()-t0>2: cv2.imwrite('/home/robot/cap_frame.jpg',f); saved=1
json.dump(out,open('/home/robot/corners.json','w')); print(len(out),'cuadros con 4 tags')
