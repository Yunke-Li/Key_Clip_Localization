import sys
import numpy as np
import cv2
from sklearn.decomposition import PCA
import copy

from utils import dense_flow


path = 'D:\\Key_Clip_Localization\\data\\Abuse001_x264.mp4'

cap = cv2.VideoCapture(path)
framecount = copy.deepcopy(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
frameWidth = copy.deepcopy(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
frameHeight = copy.deepcopy(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps =cap.get(cv2.CAP_PROP_FPS)
# #fps=30
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# #size=(960,544)
# i=0
# while(cap.isOpened()):
#     i=i+1
#     ret, frame = cap.read()
#     if ret==True:
#         cv2.imwrite('D:\\Key_Clip_Localization\\data\\Abuse001_x264\\'+str(i)+'.jpg', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
#
# cv2.destroyAllWindows()


flow_map = np.empty((frameHeight, frameWidth, 2, framecount), np.dtype('float'))
flow_pca = np.empty((frameHeight, frameWidth, 2, framecount), np.dtype('float'))
for i in range(1, framecount):
    print(i)
    previous_img = cv2.imread('D:\\Key_Clip_Localization\\data\\Abuse001_x264\\' + str(i) + '.jpg')
    next_img = cv2.imread('D:\\Key_Clip_Localization\\data\\Abuse001_x264\\' + str(i+1) + '.jpg')
    flow = dense_flow(previous_img, next_img)
    flow_reshape = flow.reshape(frameWidth*frameHeight, 2)
    pca = PCA(n_components=0.9)
    reduced_flow = pca.fit_transform(flow_reshape)
    # flow_map[:, :, :, (i-1)] = flow

