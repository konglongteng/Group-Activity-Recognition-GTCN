import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random

import os
import sys
"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
@kong: change it to volleyballtatic style
"""

# ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
#               'l_set', 'l-spike', 'l-pass', 'l_winpoint']

ACTIVITIES = ['Smash', 'Open', 'Switch', 'Space', 'Receive', 'Defense']

NUM_ACTIVITIES = 6

# ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
#            'moving', 'setting', 'spiking', 'standing',
#            'waiting']
# NUM_ACTIONS = 9


def volleytatic_read_annotations(path, annofile, sid):
    """
    reading annotations for the given sequence
    """
    annotations = {}
    detpath = '/home/kong/mydisk/taticDataset/volleyballtatic_det/'

    # gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    # act_to_id = {name: i for i, name in enumerate(ACTIONS)}
    annofile = os.path.join(path, annofile)
    video_list = [VideoRecord(x.strip().split(' ')) for x in open(annofile)]

    for vitem in video_list:
        samplepath = vitem.path
        samplepatharr = samplepath.split('/')
        if int(samplepatharr[1]) == sid:
            file_name = samplepatharr[2] #/1/10050 split to'',1,10050
            activity = vitem.label

            #read bboxes from /home/kong/mydisk/taticDataset/volleyballtatic_det
            detfile = os.path.join(detpath, samplepatharr[1], '%s_det.txt' % file_name)
            bboxeslines = []
            bboxes = []
            for line in open(detfile, "r"):
                linearr = line.split(',')
                if int(linearr[0]) == 1: #only the first frame
                    bboxes.append([int(linearr[2]),int(linearr[3]),int(linearr[4])+int(linearr[2]),int(linearr[5])+int(linearr[3])])

            bboxes = np.array(bboxes)


            fid = int(file_name)
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                # 'actions': actions,
                'bboxes': bboxes,
                'numframes':vitem.num_frames,
            }

    return annotations


# def volley_read_dataset(path, seqs):
#     data = {}
#     for sid in seqs:
#         data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
#     return data

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

#@kong for volleyball tatic
def volleytatic_read_dataset(path, annofile):
    detpath = '/home/kong/mydisk/taticDataset/volleyballtatic_detfeat/'
    annofile = os.path.join(detpath,annofile)
    data = {}
    video_list = [VideoRecord(x.strip().split(' ')) for x in open(annofile)]
    seqs = set()
    for vitem in video_list:
        samplepath = vitem.path
        samplepatharr = samplepath.split('/')
        seqs.add(int(samplepatharr[1]))


    for sid in seqs: #sid is the videoname
        data[sid] = volleytatic_read_annotations(path, annofile, sid)
    return data




def volleytatic_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid-num_before, src_fid+num_after+1)]


def load_samples_sequence(anns,tracks,images_path,frames,image_size,num_boxes=12,):
    """
    load samples of a bath
    
    Returns:
        pytorch tensors
    """
    images, boxes, boxes_idx = [], [], []
    activities, actions = [], []
    for i, (sid, src_fid, fid) in enumerate(frames):
        #img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        #img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)
        
        img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        
        img=transforms.functional.resize(img,image_size)
        img=np.array(img)
        
        # H,W,3 -> 3,H,W
        img=img.transpose(2,0,1)
        images.append(img)

        boxes.append(tracks[(sid, src_fid)][fid])
        actions.append(anns[sid][src_fid]['actions'])
        if len(boxes[-1]) != num_boxes:
          boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes-len(boxes[-1])]])
          actions[-1] = actions[-1] + actions[-1][:num_boxes-len(actions[-1])]
        boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
        activities.append(anns[sid][src_fid]['group_activity'])


    images = np.stack(images)
    activities = np.array(activities, dtype=np.int32)
    bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
    bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
    actions = np.hstack(actions).reshape([-1, num_boxes])
    
    #convert to pytorch tensor
    images=torch.from_numpy(images).float()
    bboxes=torch.from_numpy(bboxes).float()
    bboxes_idx=torch.from_numpy(bboxes_idx).int()
    actions=torch.from_numpy(actions).long()
    activities=torch.from_numpy(activities).long()

    return images, bboxes, bboxes_idx, actions, activities


class VolleytaticDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """
    def __init__(self,anns,tracks,frames,images_path,image_size,feature_size,num_boxes=12,num_before=4,num_after=4,is_training=True,is_finetune=False):
        self.anns=anns
        # self.tracks=tracks   @kong we have detections in every frame
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        
        self.num_boxes=num_boxes
        # self.num_before=num_before @kong each clip has different length
        # self.num_after=num_after
        
        self.is_training=is_training
        self.is_finetune=is_finetune
    
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        select_frames=self.volley_frames_sample(self.frames[index])
        sample=self.load_samples_sequence(select_frames)
        
        return sample
    
    def volley_frames_sample(self,frame):
        
        sid, src_fid = frame
        #@kong
        num_offrames = self.anns[sid][src_fid]['numframes']
        
        if self.is_finetune:
            if self.is_training:
                #fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)
                fid = random.randint(src_fid, src_fid + num_offrames-1)
                return [(sid, src_fid, fid)]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid, src_fid+num_offrames-1, num_offrames/15)]
        else:
            if self.is_training:
                #sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
                #@kong random 3 frames
                sample_frames = random.sample(range(src_fid, src_fid + num_offrames-1), 3)
                return [(sid, src_fid, fid) 
                        for fid in sample_frames]
            else:
                # return [(sid, src_fid, fid)
                #         for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
                #@kong
                return [(sid, src_fid, fid)
                        for fid in  [src_fid,src_fid+num_offrames/15, src_fid+num_offrames*2/15,src_fid+num_offrames*3/15,src_fid+num_offrames*4/15, src_fid+num_offrames*5/15,src_fid+num_offrames*6/15,src_fid+num_offrames*7/15,src_fid+num_offrames*8/15]]

    
    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        print('select_frames:',select_frames)
        OH, OW=self.feature_size
        
        images, boxes = [], []
        activities, actions = [], []
        for i, (sid, src_fid, fid) in enumerate(select_frames):
            # sid = 19
            # src_fid = 65964
            # fid = 66053
            #19, 65964, 66053

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            #@kong
            originalsize = img.size
            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)

            # temp_boxes=np.ones_like(self.tracks[(sid, src_fid)][fid])
            # for i,track in enumerate(self.tracks[(sid, src_fid)][fid]):
            #
            #     y1,x1,y2,x2 = track
            #     w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
            #     temp_boxes[i]=np.array([w1,h1,w2,h2])
            #
            # boxes.append(temp_boxes)
            #
            #
            # actions.append(self.anns[sid][src_fid]['actions'])

            #@kong we rewrite the bboxes in track
            # read bboxes from /home/kong/mydisk/taticDataset/volleyballtatic_det
            detpath = '/home/kong/mydisk/taticDataset/volleyballtatic_det'
            detfile = os.path.join(detpath, str(sid), '%s_det.txt' % src_fid)
            #the ratio resize image and source imgage

            temp_boxes = []
            #@kong use np contain bboxes
            temp_boxes_np = np.zeros((12,4),dtype=float)
            index_boxes = 0
            for line in open(detfile, "r"):
                linearr = line.split(',')
                if int(linearr[0]) == fid-src_fid+1:  # random frame in the clip
                    [x1, y1, x2, y2] = [float(linearr[2])/originalsize[0], float(linearr[3])/originalsize[1], (float(linearr[4]) + float(linearr[2]))/originalsize[0],
                                 (float(linearr[5]) + float(linearr[3]))/originalsize[1]]
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH

                    #@kong np
                    temp_boxes_np[index_boxes] = [w1, h1, w2, h2]
                    index_boxes = index_boxes + 1

                    # temp_boxes.append([w1, h1, w2, h2])
                    # if len(temp_boxes) >= 12:
                    #     continue

            # np_temp_boxes = np.array(temp_boxes)
            # boxes.append(np_temp_boxes)
            temp_boxes_np[index_boxes:] = temp_boxes_np[:12 - index_boxes]#change 0 to first rows
            boxes.append(temp_boxes_np)


            # if len(boxes[-1]) != self.num_boxes:
            #     boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
            #     print(boxes[-1].shape)
                # actions[-1] = actions[-1] + actions[-1][:self.num_boxes-len(actions[-1])]
            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        print(boxes)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        # actions = np.hstack(actions).reshape([-1, self.num_boxes])
        #@kong fake values
        actions = np.zeros((1, 12))
        

        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        #print('sample ok')

        return images, bboxes,  actions, activities
    
