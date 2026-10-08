import cv2, numpy as np, time, json, math
CAM='/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0918_20201224_v008-video-index0'
fs=cv2.FileStorage('/home/robot/cam_ball_balancer/src/cam_ball_balancer_cpp/config/camera_intrinsics.yml',0)
K=fs.getNode('camera_matrix').mat(); D=fs.getNode('dist_coeffs').mat()
ids_cfg=[0,1,3,2]; pos=[(-0.035,0.0415),(0.035,0.0415),(0.035,-0.0415),(-0.035,-0.0415)]; h=0.015
objs={i:np.array([[cx-h,cy+h,0],[cx+h,cy+h,0],[cx+h,cy-h,0],[cx-h,cy-h,0]]) for i,(cx,cy) in zip(ids_cfg,pos)}
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
pn=cv2.aruco.DetectorParameters_create(); ps=cv2.aruco.DetectorParameters_create(); ps.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
def tilt(r):
    R,_=cv2.Rodrigues(r); n=R[:,2]; return math.degrees(math.atan2(n[1],-n[2])), math.degrees(math.atan2(n[0],-n[2]))
cap=cv2.VideoCapture(CAM,cv2.CAP_V4L2); cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')); cap.set(3,640); cap.set(4,480)
out=[]; t0=time.time()
while time.time()-t0<18:
    ok,f=cap.read()
    if not ok: continue
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); row={'t':time.time()-t0}
    for name,p in (('none',pn),('subpix',ps)):
        ts=time.perf_counter(); c,ids,_=cv2.aruco.detectMarkers(g,d,parameters=p); row[name+'_ms']=(time.perf_counter()-ts)*1e3
        if ids is None: continue
        O=[];I=[]
        for cc,i in zip(c,ids.ravel()):
            if int(i) in objs: O.append(objs[int(i)]); I.append(cc.reshape(4,2))
        if len(O)<2: continue
        n,rv,tv,err=cv2.solvePnPGeneric(np.concatenate(O),np.concatenate(I).astype(np.float64),K,D,flags=cv2.SOLVEPNP_IPPE)
        e=err.ravel(); o=np.argsort(e)
        row[name]={'e':[float(e[o[0]]),float(e[o[1]]) if n>1 else None],'best':tilt(rv[o[0]]),'alt':tilt(rv[o[1]]) if n>1 else None,'nt':len(O)}
    out.append(row)
json.dump(out,open('/home/robot/diag_ippe.json','w')); print(len(out),'cuadros')
