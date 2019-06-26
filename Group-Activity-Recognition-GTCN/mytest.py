import time
import os
import random
import numpy as np
#os.mkdir('result/[Volleyball_stage1_stage1]<2019-06-11_09-59-09>')
# for i in range(0, 5*12, 5):
#     print(i)
#
[x1,y1,x2,y2] = [1,2,3,4]
a = [x1,y1,x2,y2]
#print([x1,y1,x2,y2])
# print(a[-1])
sid = 1
src_fid = 2

#print([(sid, src_fid, fid) for fid in range(src_fid-4, src_fid+5+1,3)])
#print(random.sample(range(src_fid, src_fid + 6), 3))

# print([(sid, src_fid, fid)
#  for fid in
#  [src_fid - 3, src_fid, src_fid + 3, src_fid - 4, src_fid - 1, src_fid + 2, src_fid - 2, src_fid + 1, src_fid + 4]])
bboxeslist = []
boxes = np.array([[ 67.54270833,  35.28333333,  81.36197917,  54.29444444],
       [131.978125  ,  36.57222222, 143.2625    ,  62.83333333],
       [ 90.43854167,  35.60555556, 102.37708333,  59.20833333],
       [ 37.0421875 ,  28.11388889,  46.28229167,  45.51388889],
       [  5.396875  ,  36.65277778,  18.153125  ,  60.175     ],
       [ 80.953125  ,  26.98611111,  88.47604167,  44.78888889],
       [ 14.14635417,  50.75      ,  34.50729167,  74.83611111],
       [ 55.76770833,  26.42222222,  65.82552083,  44.95      ],
       [ 46.93645833,  24.24722222,  55.03177083,  40.92222222],
       [ 66.4796875 ,  30.04722222,  73.92083333,  47.44722222],
       [ 0,  0,  0,  0],
       [0  ,  0, 0   , 0]])

# bboxeslist.append(boxes)
#
boxes[10:] = boxes[:2]
# bboxes = np.vstack(bboxeslist).reshape([-1, 12, 4])
# print(boxes[10:])
print(1/15)