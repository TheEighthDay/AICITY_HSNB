import os 
import json
import numpy as np 
import pandas as pd  
import torch
import pickle
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--results_128_8', type=str)
parser.add_argument('--results_128_32', type=str)
parser.add_argument('--results_32_8', type=str)
parser.add_argument('--results_right_128_32', type=str)
parser.add_argument('--path_to_fea_right_128_32', type=str)
parser.add_argument('--path_to_fea_128_8', type=str)
parser.add_argument('--path_to_fea_128_32', type=str)
parser.add_argument('--path_to_fea_32_8', type=str)

args = parser.parse_args()

video_idx_map={"Dashboard_user_id_42271_NoAudio_3":"1","Dashboard_user_id_42271_NoAudio_4":"2","Dashboard_user_id_56306_NoAudio_2":"3",\
    "Dashboard_user_id_56306_NoAudio_3":"4","Dashboard_User_id_65818_NoAudio_1":"5","Dashboard_User_id_65818_NoAudio_2":"6",\
    "Dashboard_User_id_72519_NoAudio_2":"7","Dashboard_User_id_72519_NoAudio_3":"8","Dashboard_User_id_79336_NoAudio_0":"9","Dashboard_User_id_79336_NoAudio_2":"10"}

video_idx_map_right={"Right_side_window_user_id_42271_NoAudio_3":"1","Right_side_window_user_id_42271_NoAudio_4":"2","Rightside_window_user_id_56306_NoAudio_2":"3",\
    "Rightside_window_user_id_56306_NoAudio_3":"4","Rightside_window_User_id_65818_NoAudio_1":"5","Rightside_window_User_id_65818_NoAudio_2":"6",\
    "Right_window_User_id_72519_NoAudio_2":"7","Right_window_User_id_72519_NoAudio_3":"8","Rightside_window_User_id_79336_NoAudio_0":"9","Rightside_window_User_id_79336_NoAudio_2":"10"}

video_idx_map_new = {}
for k,v in video_idx_map.items():
    video_idx_map_new["_".join(k.split("_")[-3:])]=v
video_idx_map = video_idx_map_new 


label_min_th_map = {0:0,1:0.95,2:0.85,3:0.41,4:0.55,5:0.79,6:0.88,7:0.85 \
                    ,8:0.85,9:0.29,10:0.89,11:0.53,12:0.63,13:0.23,14:0.75,15:0.53,16:0,17:0.92}

label_mean_th_map = {0:0,1:0.97,2:0.90,3:0.73,4:0.65,5:0.89,6:0.88,7:0.90 \
                    ,8:0.95,9:0.75,10:0.90,11:0,12:0.82,13:0.63,14:0.89,15:0.79,16:0,17:0.93}

label_newmean_th_map = {0:0,1:0.91,2:0.87,3:0.71,4:0.60,5:0.82,6:0.68,7:0.92 \
                    ,8:0.81,9:0.77,10:0.74,11:0.75,12:0.73,13:0.67,14:0.75,15:0.76,16:0.66,17:0.75}

label_newmin_th_map = {0:0,1:0.91,2:0.85,3:0.41,4:0.60,5:0.79,6:0.88,7:0.90 \
                    ,8:0.81,9:0.75,10:0.89,11:0.75,12:0.63,13:0.23,14:0.75,15:0.58,16:0.05,17:0.75}

iou_th_map = {1:0.31,\
2: 0.24, \
3: 0.26, \
4: 0.15, \
5: 0.13, \
6: 0.22, \
7: 0.28, \
8: 0.25, \
9: 0.24, \
10: 0.12, \
11: 0.18, \
12: 0.24, \
13: 0.27, \
14: 0.27, \
15: 0.22, \
16: 0.16, \
17: 0.17}

videoindex_list = []
class_list = []

class message(object):
    def __init__(self,videox,startx,endx,scorex,labelx,predx,label_score,index):
        self.video_id = video_idx_map["_".join(videox.split("_")[-3:])]
        self.start = int(startx)
        self.end = int(endx)
        self.label = int(labelx)
        self.pred = predx
        self.iou_score = scorex
        self.label_score = label_score
        self.index = index

    def __lt__(self, other):
        return self.iou_score < other.iou_score


for i in range(11):
    temp = []
    for j in range(18):
        t = []
        temp.append(t)
    class_list.append(temp)

for i in range(11):
    temp = []
    videoindex_list.append(temp)

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def post(video,start,end,score,pred):
    ret_video=[]
    ret_start=[]
    ret_end=[]
    ret_score=[]
    ret_label=[]
    ret_label_score=[]
    ret_index=[]
    label_list = ["Dashboard_user_id_42271_NoAudio_3","Dashboard_user_id_42271_NoAudio_4","Dashboard_user_id_56306_NoAudio_2","Dashboard_user_id_56306_NoAudio_3","Dashboard_User_id_65818_NoAudio_1","Dashboard_User_id_65818_NoAudio_2"]
    # print(score)
    # print("video",len(video))
    if_cnt = 0
    else_cnt = 0
    for i,(videox,startx,endx,scorex,predx) in enumerate(zip(video,start,end,score,pred)):
        # print("predx",predx)
        # print("predx",np.max(predx))
        # pred1 = np.argmax(predx)
        # print("pred1",pred1)
        # input()
        # c = np.argsort(pred,1)
        # for j in range(len(pred1)):
        #     if pred1[j] == 0:
        #         for i in range(18):
        #             # print(c[j][i])
        #             if c[j][i] == 16:
        #                 pred1[j] = i
        #                 break
        #scorex>0.13 0.9
        # and (np.max(predx)>0.1)
        # print("i ",i)
        if(endx-startx>10 and endx-startx<30):
                
            # if (videox=="Dashboard_user_id_42271_NoAudio_3" and int(startx) not in range(435,467) and int(endx) not in range(435,467)) \
            # or (videox=="Dashboard_user_id_42271_NoAudio_4" and int(startx) not in range(443,475) and int(endx) not in range(443,475)) \
            # or (videox=="Dashboard_user_id_56306_NoAudio_2" and int(startx) not in range(157,198) and int(endx) not in range(157,198)) \
            # or (videox=="Dashboard_user_id_56306_NoAudio_3" and int(startx) not in range(154,194) and int(endx) not in range(154,194)) \
            # or (videox=="Dashboard_User_id_65818_NoAudio_1" and int(startx) not in range(297,350) and int(endx) not in range(297,350)) \
            # or (videox=="Dashboard_User_id_65818_NoAudio_2" and int(startx) not in range(321,405) and int(endx) not in range(321,405)) : 
                if np.argmax(predx) == 0:
                    c = np.argsort(predx)
                    labelx = c[-2]
                    if predx[c[-2]] > label_min_th_map[labelx]:
                        ret_label_score.append(predx[c[-2]])
                        a = message(videox,startx,endx,scorex,labelx,predx,predx[c[-2]],i)
                        videoindex_list[int(video_idx_map["_".join(videox.split("_")[-3:])])].append(a)
                        ret_index.append(i)
                        ret_video.append(videox)
                        ret_start.append(startx)
                        ret_end.append(endx)
                        ret_score.append(scorex)
                        ret_label.append(labelx)
                        if_cnt += 1
                    # print("a.label",a.label)
                else:
                    labelx = np.argmax(predx)
                    if np.max(predx) > label_min_th_map[labelx]:
                        ret_label_score.append(np.max(predx))
                        a = message(videox,startx,endx,scorex,labelx,predx,np.max(predx),i)
                        videoindex_list[int(video_idx_map["_".join(videox.split("_")[-3:])])].append(a)
                        ret_index.append(i)
                        ret_video.append(videox)
                        ret_start.append(startx)
                        ret_end.append(endx)
                        ret_score.append(scorex)
                        ret_label.append(labelx) 
                        else_cnt += 1       
            # elif videox not in label_list:
            #     if np.argmax(predx) == 0:
            #         for i in range(18):
            #             if predx[i] == 16:
            #                 labelx = i
            #                 break
            #     else:
            #         labelx = np.argmax(predx)
            #     ret_video.append(videox)
            #     ret_start.append(startx)
            #     ret_end.append(endx)
            #     ret_score.append(scorex)
            #     ret_label.append(labelx)
    # print("ret_video",len(ret_video))
    # print("if_cnt",if_cnt)
    # print("else_cnt",else_cnt)
    return ret_video,ret_start,ret_end,ret_score,ret_label_score,ret_label,ret_index

def get_label(json_path,pred_path,video,start,end):
    start_frame = []
    end_frame = []

    gt_data = json.load(open(json_path,"rb"))

    for pv,ps,pe in zip(video,start,end):
        duration = gt_data["database"][pv]["duration"]
        duration_frame = gt_data["database"][pv]["duration_frame"]
        start_frame.append(int(ps/duration*duration_frame))
        end_frame.append(int(pe/duration*duration_frame))

    pred_data = {}
    for pv in video:
        pred_video_path = os.path.join(pred_path,pv)
        if pred_video_path not in pred_data:
            pred_data[pv] = np.load(pred_video_path+"_pred.npy")


    start_snippet = [ max(int((x - 12) // 8),0) for x in start_frame]
    end_snippet = [ max(int((x - 12) // 8),0)+1 for x in end_frame]

    label=[]

    for pv,ss,es in zip(video,start_snippet,end_snippet):
        preds = pred_data[pv][ss:es]
        preds = torch.softmax(torch.tensor(preds),1).numpy()
        preds =np.mean(preds,axis=0)
        label.append(np.argmax(preds))
    return label

def gen_submission(pkl_path,json_path,pred_path):

    video_idx_map={"Dashboard_user_id_42271_NoAudio_3":"1","Dashboard_user_id_42271_NoAudio_4":"2","Dashboard_user_id_56306_NoAudio_2":"3",\
    "Dashboard_user_id_56306_NoAudio_3":"4","Dashboard_User_id_65818_NoAudio_1":"5","Dashboard_User_id_65818_NoAudio_2":"6",\
    "Dashboard_User_id_72519_NoAudio_2":"7","Dashboard_User_id_72519_NoAudio_3":"8","Dashboard_User_id_79336_NoAudio_0":"9","Dashboard_User_id_79336_NoAudio_2":"10"}

    video_idx_map_new = {}
    for k,v in video_idx_map.items():
        video_idx_map_new["_".join(k.split("_")[-3:])]=v
    video_idx_map = video_idx_map_new 


    f=open(pkl_path,"r")
    data = f.readlines()
    f.close()
    
    video = [x.split(" ")[0] for x in data]
    start = [float(x.split(" ")[1]) for x in data]
    end = [float(x.split(" ")[2]) for x in data]
    score = [float(x.split(" ")[4]) for x in data]

    # print(len(video))
    label = get_label(json_path,pred_path,video,start,end)

    video,start,end,score,label = post(video,start,end,score,label)

    f=open("submission.csv","w")
    for video_id,activity_id,start_time,end_time in zip(video,label,start,end):
        f.write("{} {} {} {}\n".format(video_idx_map["_".join(video_id.split("_")[-3:])],activity_id,int(start_time),int(end_time)))
    f.close()


def get_label_2view(cla_path):
    #/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_proposal/dashboard_sparefeat_A2
    dash_view_pred_path  = os.path.join(cla_path,"dash_fea_32x256x1x3.pkl")
    right_view_pred_path  = os.path.join(cla_path,"right_fea_32x256x1x3.pkl")
    dash_view_pred  =  pickle.load(open(dash_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred  =  pickle.load(open(right_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    pred = (dash_view_pred+right_view_pred)/2
    pred = np.argmax(pred,1)
    return [x for x in pred]


def get_label_3view(cla_path):
    #/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_proposal/dashboard_sparefeat_A2
    dash_view_pred_path  = os.path.join(cla_path,"dash_fea_32x256x1x3.pkl")
    right_view_pred_path  = os.path.join(cla_path,"right_fea_32x256x1x3.pkl")
    rear_view_pred_path  = os.path.join(cla_path,"rear_fea_32x256x1x3.pkl")
    dash_view_pred  =  pickle.load(open(dash_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred  =  pickle.load(open(right_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred  =  pickle.load(open(rear_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    pred = (dash_view_pred+right_view_pred+rear_view_pred)/3
    pred1 = np.argmax(pred,1)
    c = np.argsort(pred,1)
    for j in range(len(pred1)):
        if pred1[j] == 0:
            for i in range(18):
                # print(c[j][i])
                if c[j][i] == 16:
                    pred1[j] = i
                    break
    return [x for x in pred1]

def get_label_multi_3view(cla_path):
    #/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_proposal/dashboard_sparefeat_A2
    dash_view_pred_path_64  = os.path.join(cla_path,"dash_fea_64x256x1x3.pkl")
    right_view_pred_path_64  = os.path.join(cla_path,"right_fea_64x256x1x3.pkl")
    rear_view_pred_path_64  = os.path.join(cla_path,"rear_fea_64x256x1x3.pkl")
    dash_view_pred_path_128  = os.path.join(cla_path,"dash_fea_128x256x1x3.pkl")
    right_view_pred_path_128  = os.path.join(cla_path,"right_fea_128x256x1x3.pkl")
    rear_view_pred_path_128  = os.path.join(cla_path,"rear_fea_128x256x1x3.pkl")
    dash_view_pred_path_32  = os.path.join(cla_path,"dash_fea_32x256x1x3.pkl")
    right_view_pred_path_32  = os.path.join(cla_path,"right_fea_32x256x1x3.pkl")
    rear_view_pred_path_32  = os.path.join(cla_path,"rear_fea_32x256x1x3.pkl")
    dash_view_pred_64  =  pickle.load(open(dash_view_pred_path_64,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred_64  =  pickle.load(open(right_view_pred_path_64,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred_64  =  pickle.load(open(rear_view_pred_path_64,"rb"))['video_preds'].softmax(1).numpy()
    dash_view_pred_128  =  pickle.load(open(dash_view_pred_path_128,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred_128  =  pickle.load(open(right_view_pred_path_128,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred_128  =  pickle.load(open(rear_view_pred_path_128,"rb"))['video_preds'].softmax(1).numpy()
    dash_view_pred_32  =  pickle.load(open(dash_view_pred_path_32,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred_32  =  pickle.load(open(right_view_pred_path_32,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred_32  =  pickle.load(open(rear_view_pred_path_32,"rb"))['video_preds'].softmax(1).numpy()
    pred_32 = (dash_view_pred_32+right_view_pred_32+rear_view_pred_32)/3
    pred_64 = (dash_view_pred_64+right_view_pred_64+rear_view_pred_64)/3
    pred_128 = (dash_view_pred_128+right_view_pred_128+rear_view_pred_128)/3
    pred = (pred_32 + pred_64 + pred_128)/3
    pred3 = pred_32*0.25 + pred_64*0.25 + pred_128*0.5
    pred32_128 = pred_32*0.25 + pred_128*0.75
    pred64_128 = pred_64*0.25 + pred_128*0.75
    pred = pred32_128
    pred1 = np.argmax(pred,1)
    c = np.argsort(pred,1)
    for j in range(len(pred1)):
        if pred1[j] == 0:
            for i in range(18):
                # print(c[j][i])
                if c[j][i] == 16:
                    pred1[j] = i
                    break
    return [x for x in pred1]

def get_pred_3view(cla_path):
    #/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_proposal/dashboard_sparefeat_A2
    dash_view_pred_path  = os.path.join(cla_path,"dash_fea_32x256x1x3.pkl")
    right_view_pred_path  = os.path.join(cla_path,"right_fea_32x256x1x3.pkl")
    rear_view_pred_path  = os.path.join(cla_path,"rear_fea_32x256x1x3.pkl")
    dash_view_pred  =  pickle.load(open(dash_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred  =  pickle.load(open(right_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred  =  pickle.load(open(rear_view_pred_path,"rb"))['video_preds'].softmax(1).numpy()
    pred = (dash_view_pred+right_view_pred+rear_view_pred)/3
    return pred

def get_pred_multi_3view(cla_path):
    #/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_proposal/dashboard_sparefeat_A2
    dash_view_pred_path_64  = os.path.join(cla_path,"dash_fea_64x256x1x3.pkl")
    right_view_pred_path_64  = os.path.join(cla_path,"right_fea_64x256x1x3.pkl")
    rear_view_pred_path_64  = os.path.join(cla_path,"rear_fea_64x256x1x3.pkl")
    dash_view_pred_path_128  = os.path.join(cla_path,"dash_fea_128x256x1x3.pkl")
    right_view_pred_path_128  = os.path.join(cla_path,"right_fea_128x256x1x3.pkl")
    rear_view_pred_path_128  = os.path.join(cla_path,"rear_fea_128x256x1x3.pkl")
    dash_view_pred_path_32  = os.path.join(cla_path,"dash_fea_32x256x1x3.pkl")
    right_view_pred_path_32  = os.path.join(cla_path,"right_fea_32x256x1x3.pkl")
    rear_view_pred_path_32  = os.path.join(cla_path,"rear_fea_32x256x1x3.pkl")
    dash_view_pred_64  =  pickle.load(open(dash_view_pred_path_64,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred_64  =  pickle.load(open(right_view_pred_path_64,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred_64  =  pickle.load(open(rear_view_pred_path_64,"rb"))['video_preds'].softmax(1).numpy()
    dash_view_pred_128  =  pickle.load(open(dash_view_pred_path_128,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred_128  =  pickle.load(open(right_view_pred_path_128,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred_128  =  pickle.load(open(rear_view_pred_path_128,"rb"))['video_preds'].softmax(1).numpy()
    dash_view_pred_32  =  pickle.load(open(dash_view_pred_path_32,"rb"))['video_preds'].softmax(1).numpy()
    right_view_pred_32  =  pickle.load(open(right_view_pred_path_32,"rb"))['video_preds'].softmax(1).numpy()
    rear_view_pred_32  =  pickle.load(open(rear_view_pred_path_32,"rb"))['video_preds'].softmax(1).numpy()
    pred_32 = dash_view_pred_32*0.9+right_view_pred_32*0.05+rear_view_pred_32*0.05
    pred_64 = dash_view_pred_64*0.9+right_view_pred_64*0.05+rear_view_pred_64*0.05
    pred_128 = dash_view_pred_128*0.9+right_view_pred_128*0.05+rear_view_pred_128*0.05
    # pred_32 = dash_view_pred_32*0.6+right_view_pred_32*0.2+rear_view_pred_32*0.2
    # pred_64 = dash_view_pred_64*0.6+right_view_pred_64*0.2+rear_view_pred_64*0.2
    # pred_128 = dash_view_pred_128*0.6+right_view_pred_128*0.2+rear_view_pred_128*0.2
    pred = (pred_32 + pred_64 + pred_128)/3
    pred3 = pred_32*0.25 + pred_64*0.25 + pred_128*0.5
    pred32_128 = pred_32*0.25 + pred_128*0.75
    pred64_128 = pred_64*0.3 + pred_128*0.7
    pred = pred3
    return pred
    
def get_pred_super(pred1,pred2,pred3):
    #/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_proposal/dashboard_sparefeat_A2
    # pred = (pred1+pred2+pred3)/3
    pred = pred1*0.2+pred2*0.2+pred3*0.6
    return pred

def gen_submission_2view(pkl_path,cla_path1,cla_path2,cla_path3):

    video_idx_map={"Dashboard_user_id_42271_NoAudio_3":"1","Dashboard_user_id_42271_NoAudio_4":"2","Dashboard_user_id_56306_NoAudio_2":"3",\
    "Dashboard_user_id_56306_NoAudio_3":"4","Dashboard_User_id_65818_NoAudio_1":"5","Dashboard_User_id_65818_NoAudio_2":"6",\
    "Dashboard_User_id_72519_NoAudio_2":"7","Dashboard_User_id_72519_NoAudio_3":"8","Dashboard_User_id_79336_NoAudio_0":"9","Dashboard_User_id_79336_NoAudio_2":"10"}

    video_idx_map_new = {}
    for k,v in video_idx_map.items():
        video_idx_map_new["_".join(k.split("_")[-3:])]=v
    video_idx_map = video_idx_map_new 


    pkl_path = args.results_32_8#"eval_results.pkl"
    f=open(pkl_path,"r")
    data = f.readlines()
    f.close()
    # print("data",len(data),data[2])
    video1 = [x.split(" ")[0] for x in data]
    start1 = [float(x.split(" ")[1]) for x in data]
    end1 = [float(x.split(" ")[2]) for x in data]
    score1 = [float(x.split(" ")[4]) for x in data]

    pkl_path = args.results_128_8#"128_8_eval_results.pkl"
    f=open(pkl_path,"r")
    data = f.readlines()
    f.close()
    # print("data",len(data),data[2])
    video2 = [x.split(" ")[0] for x in data]
    start2 = [float(x.split(" ")[1]) for x in data]
    end2 = [float(x.split(" ")[2]) for x in data]
    score2 = [float(x.split(" ")[4]) for x in data]

    pkl_path = args.results_128_32#"128_32_eval_results.pkl"
    f=open(pkl_path,"r")
    data = f.readlines()
    f.close()
    # print("data",len(data),data[2])
    video3 = [x.split(" ")[0] for x in data]
    start3 = [float(x.split(" ")[1]) for x in data]
    end3 = [float(x.split(" ")[2]) for x in data]
    score3 = [float(x.split(" ")[4]) for x in data]

    pkl_path = args.results_right_128_32#"right_128_32_eval_results.pkl"
    f=open(pkl_path,"r")
    data = f.readlines()
    f.close()
    # print("data",len(data),data[2])
    video4 = [x.split(" ")[0] for x in data]
    start4 = [float(x.split(" ")[1]) for x in data]
    end4 = [float(x.split(" ")[2]) for x in data]
    score4 = [float(x.split(" ")[4]) for x in data]

    # print(len(video))
    # label = get_label_multi_3view(cla_path1)
    pred1 = get_pred_multi_3view(cla_path1)
    pred2 = get_pred_multi_3view(cla_path2)
    pred3 = get_pred_multi_3view(cla_path3)
    cla_path4 =args.path_to_fea_right_128_32#"./right_128_for128_32_A2"
    pred4 = get_pred_multi_3view(cla_path4)
    # pred = get_pred_super(pred1,pred2,pred3)
    # pred = pred1
    # print("label",len(label),label[0])
    # print("label[2001:4001]",label[2001:4001])

    video,start,end,iou_score,label_score,label,index = post(video1,start1,end1,score1,pred1)
    # print(len(videoindex_list[2]))
    video,start,end,iou_score,label_score,label,index = post(video2,start2,end2,score2,pred2)
    # print(len(videoindex_list[2]))
    video,start,end,iou_score,label_score,label,index = post(video3,start3,end3,score3,pred3)

    # video,start,end,iou_score,label_score,label,index = post(video4,start4,end4,score4,pred4)

    # print(len(videoindex_list[2]))
    for i in range(11):
        if i!=0:
            for j in range(len(videoindex_list[i])):
                # print("i,j,id",i,j,videoindex_list[i][j].video_id)
                class_list[i][videoindex_list[i][j].label].append(videoindex_list[i][j])

    k_list = []
    for i in range(11):
        if i!=0:
            for j in range(len(class_list[i])):
                if class_list[i][j] != []:
                    # print("----j----",j)
                    class_list[i][j].sort()
                    if class_list[i][j][-1].iou_score > iou_th_map[int(class_list[i][j][-1].label)]:
                        if len(class_list[i][j])>=2:
                            if (class_list[i][j][-1].iou_score>0.25) and (class_list[i][j][-2].iou_score>0.25) and \
                                (abs(class_list[i][j][-1].start - class_list[i][j][-2].start)<=2) and \
                                (abs(class_list[i][j][-1].end  - class_list[i][j][-2].end)<=2) and \
                                (class_list[i][j][-1].label == class_list[i][j][-2].label):
                                # print("----")
                                k_list.append("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int((class_list[i][j][-1].start+class_list[i][j][-2].start)/2),int((class_list[i][j][-1].end+class_list[i][j][-2].end)/2)))                    
                            else:
                                if class_list[i][j][-1].iou_score>0.25:
                                    k_list.append("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
                                    # print("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
                                
                                if class_list[i][j][-2].iou_score>0.25:
                                    k_list.append("{} {} {} {}".format(int(class_list[i][j][-2].video_id),class_list[i][j][-2].label,int(class_list[i][j][-2].start),int(class_list[i][j][-2].end)))
                                    # print("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
                                    # print("class_list[i][j][-2]",class_list[i][j][-2].video_id)
                        else: 
                            if class_list[i][j][-1].iou_score>0.25:
                                k_list.append("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
                                # print("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
                                # print("else",class_list[i][j][-1].video_id)

    # k_list = []
    # for i in range(11):
    #     if i!=0:
    #         for j in range(len(class_list[i])):
    #             if class_list[i][j] != []:
    #                 # print("----j----",j)
    #                 class_list[i][j].sort()
    #                 if len(class_list[i][j])>=2:
    #                     if class_list[i][j][-1].iou_score>0.25 and class_list[i][j][-2].iou_score>0.25:
    #                         if abs(class_list[i][j][-1].start - class_list[i][j][-2].start)<=2 and abs(class_list[i][j][-1].end  - class_list[i][j][-2].end)<=2 and class_list[i][j][-1].label == class_list[i][j][-2].label:
    #                             k_list.append("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int((class_list[i][j][-1].start+class_list[i][j][-2].start)/2),int((class_list[i][j][-1].end+class_list[i][j][-2].end)/2))
    #                         else:
    #                             k_list.append("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
    #                             k_list.append("{} {} {} {}".format(int(class_list[i][j][-2].video_id),class_list[i][j][-2].label,int(class_list[i][j][-2].start),int(class_list[i][j][-2].end)))
    #                         # print("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
    #                         # print("class_list[i][j][-2]",class_list[i][j][-2].video_id)
    #                 else:
    #                     if class_list[i][j][-1].iou_score>0.25:
    #                         # k_list.append("{} {} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end),class_list[i][j][-1].label_score))
    #                         k_list.append("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
    #                         # print("{} {} {} {}".format(int(class_list[i][j][-1].video_id),class_list[i][j][-1].label,int(class_list[i][j][-1].start),int(class_list[i][j][-1].end)))
    #                         # print("else",class_list[i][j][-1].video_id)



    # sub_list = []
    # for video_id,activity_id,start_time,end_time in zip(video,label,start,end):
    #     sub_list.append("{} {} {} {}".format(video_idx_map["_".join(video_id.split("_")[-3:])],activity_id,int(start_time),int(end_time)))
    print(len(k_list))
    # input()
    
    # f=open("jj_submission.txt","w")
    # for video_id,activity_id,start_time,end_time in zip(video,label,start,end):
    #     f.write("{} {} {} {}\n".format(video_idx_map["_".join(video_id.split("_")[-3:])],activity_id,int(start_time),int(end_time)))
    # f.close()

    f=open("submission.txt","w")
    for i in k_list:
        f.write(i+"\n")
    f.close()
    return k_list
  

def eval_iou(pkl_path,json_path):
    """
    計算每一個候選框与18个GT框的tiou，取最大的tiou作为当前候选框的tiou。
    返回所有候选框的tiou均值。
    pkl_path :eval_results.pkl
    json_path :aicity_dashboard.json
    """

    f=open(pkl_path,"r")
    pred_data = f.readlines()
    f.close()
    pred_video = [x.split(" ")[0] for x in  pred_data]
    pred_start = [float(x.split(" ")[1]) for x in  pred_data]
    pred_end = [float(x.split(" ")[2]) for x in  pred_data]

    gt_data = json.load(open(json_path,"rb"))
    mtIOU = []
    for pv,ps,pe in zip(pred_video,pred_start,pred_end):
        gt_anno = gt_data["database"][pv]["annotations"]
        gt_segments=[x["segment"] for x in gt_anno]
        gt_segments = np.array(gt_segments)
        pred_segment = np.array([ps,pe])
        tIOU=segment_iou(pred_segment,gt_segments)
        tIOU = np.max(tIOU)
        mtIOU.append(tIOU)
    mtIOU=np.array(mtIOU)
    # print(mtIOU[np.argsort(mtIOU)])
    # print(np.mean(mtIOU))

    num = len(mtIOU)
    print('mean IoU:', np.mean(mtIOU))
    print('>0.1 IoU:', (np.array(mtIOU) > 0.1).sum() / num)
    print('>0.3 IoU:', (np.array(mtIOU) > 0.3).sum() / num)
    print('>0.5 IoU:', (np.array(mtIOU) > 0.5).sum() / num)
    print('>0.7 IoU:', (np.array(mtIOU) > 0.7).sum() / num)
    print('>0.9 IoU:', (np.array(mtIOU) > 0.9).sum() / num)




if __name__=="__main__":
    
    pkl_path = args.results_32_8#s"./eval_results.pkl"
    cla_path1 =args.path_to_fea_32_8#"./dashboard_sparefeat_A2"
    cla_path2 =args.path_to_fea_128_8#"./dashboard_128_for128_8_A2"
    cla_path3 =args.path_to_fea_128_32#"./dashboard_128_for128_32_A2"
    # json_path = "/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/annotations/aicity_dashboard.json"
    # pred_path = "/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/sf_preds"
    sub_list = gen_submission_2view(pkl_path,cla_path1,cla_path2,cla_path3)

    # pkl_path = "/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/exp/dashboard_try_val/ckpt/val_eval_results.pkl"
    # pkl_path = "/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/exp/view3_try_val/ckpt/val_eval_results.pkl"
    # pkl_path = "/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/exp/dashboard_try_dense_val/ckpt/val_eval_results.pkl"
    # json_path = "/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/annotations/aicity_dashboard.json"
    # eval_iou(pkl_path,json_path)
