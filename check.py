import numpy as np
import cv2
import pykitti

basedir = './dataset/'
date = '2011_09_26'
drive = '0093'
dataset = pykitti.raw(basedir, date, drive)

from stuff import Camera, VisualOdometry

camera = Camera(1242.0, 375.0, 7.215377e+02, 7.215377e+02, 6.095593e+02,  1.728540e+02)
vo = VisualOdometry(camera, dataset)

traj = np.zeros((600,600,3), dtype=np.uint8)

for img_id in range(4541):
    img = np.array(dataset.get_gray(img_id)[0])
   
    vo.update(img, img_id)

    cur_t = vo.cur_t
    if(img_id > 2):
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
	
    else:
        x, y, z = 0., 0., 0.
    draw_x, draw_y = int(-x)+290, int(z)+90
    true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90
    print(draw_x, draw_y)
    cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1) 
    #cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2) 
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1) 
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)

cv2.imwrite('map.png', traj)
    