import cv2


def dense_flow(prvs_img, next_img):
    prvs = cv2.cvtColor(prvs_img, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev=prvs, next=next, flow=None, pyr_scale=0.5, levels=5,
                                        winsize=15,
                                        iterations=3, poly_n=3, poly_sigma=1.2,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow
