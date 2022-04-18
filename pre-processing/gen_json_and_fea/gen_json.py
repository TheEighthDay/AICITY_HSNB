# from iopath.common.file_io import g_pathmgr
from PIL import Image, ImageOps
import os
import PIL
import decord
from decord import VideoReader
from decord import cpu
from moviepy.editor import VideoFileClip
import pickle
import json

window_size = 150
overlap = 0.75
window_stride = window_size*(1-overlap)
num_frame = 32
overlap_frame = 0.75
frame_stride = num_frame*(1-overlap_frame)

import csv
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--A1_frame_path', type=str)
parser.add_argument('--A2_frame_path', type=str)

args = parser.parse_args()


def get_file_dict(path):
    file_dict = {}
    for p1 in os.listdir(path):
        p2 = os.path.join(path, p1)
        if os.path.isdir(p2):
            for p3 in os.listdir(p2):
                final_path = os.path.join(p2, p3)
                cur_path = os.path.join(p1, p3)
                file_dict[cur_path] = len(os.listdir(final_path))
    return file_dict

def video_frame(trainval=True):
    frame = 32
    stride = 8
    if trainval:
        path = args.A1_frame_path#'/mnt/lustre/share_data/shangjingjie1/AiCityClip/A1_frame'
        # prefix = '/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/extract_split'
    else:
        path = args.A2_frame_path#'/mnt/lustre/share_data/shangjingjie1/AiCityClip/A2'
        # prefix = '/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/extract_split_A2'
    file_tmpl = 'all_{}.csv'

    view_list = ['dashboard', 'right', 'rear']
    # file_list = []
    # for v in view_list:
    #     file_list.append(open(os.path.join(prefix, file_tmpl.format(v)), 'w'))

    file_dict = get_file_dict(path)

    ret_dict={}

    for path, length in file_dict.items():
        # print(path, length-frame)
        end = ""
        for i, view in enumerate(view_list):
            if view in path.lower():
                for j in range(0, length-frame, stride):
                    start = str(j)
                    end = str(j + frame)
        ret_dict[path.split("/")[1]] = int(end)
        print(path,end)
        #             # file_list[i].writelines(','.join([path, start, end]) + '\n')
    return ret_dict


def gen_json(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset):
    database={}
    for(videox,labelx,label_idx,startx,endx,startFramex,endFramex,frame_durationx,time_durationx,fpsx,feature_dimx,subsetx) in zip(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset):
        if videox not in database:
            database[videox]={}
            database[videox]["subset"]=subsetx
            database[videox]["duration"]=time_durationx
            database[videox]["duration_frame"]=frame_durationx
            database[videox]["fps"]=fpsx
            database[videox]["feature_dim"]=feature_dimx
            database[videox]["annotations"]= []
            database[videox]["annotations"].append({"label":labelx,"segment":[startx,endx],"segment(frames)":[startFramex,endFramex],"label_id":label_idx})
        else:
            database[videox]["annotations"].append({"label":labelx,"segment":[startx,endx],"segment(frames)":[startFramex,endFramex],"label_id":label_idx})
    return database

def gen_json_kc(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset):
    database={}
    for(videox,labelx,label_idx,startx,endx,startFramex,endFramex,frame_durationx,time_durationx,fpsx,feature_dimx,subsetx) in zip(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset):
        if videox not in database:
            database[videox]={}
            database[videox]["subset"]=subsetx
            database[videox]["duration_second"]=time_durationx
            database[videox]["duration_frame"]=frame_durationx
            database[videox]["fps"]=fpsx
            database[videox]["feature_dim"]=feature_dimx
            database[videox]["annotations"]= []
            database[videox]["annotations"].append({"class":labelx,"segment":[startx,endx],"segment_frame":[startFramex,endFramex],"label":label_idx})
        else:
            database[videox]["annotations"].append({"class":labelx,"segment":[startx,endx],"segment_frame":[startFramex,endFramex],"label":label_idx})
    return database



def turn_data_format(jjcsv_path,splitset="train"):  # view dash rear right

    id2label={0:"Normal Forward Driving",1:"Drinking",2:"Phone Call(right)",3:"Phone Call(left)",4:"Eating",5:"Text (Right)",\
    6:"Text (Left)",7:"Hair / makeup",8:"Reaching behind",9:"Adjust control panel",10:"Pick up from floor (Driver)",\
    11:"Pick up from floor (Passenger)",12:"Talk to passenger at the right",13:"Talk to passenger at backseat",14:"yawning",\
    15:"Hand on head",16:"Singing with music",17:"shaking or dancing with music"}

    f=open(jjcsv_path)
    data = f.readlines()
    data = data[1:]
    f.close()

    video = [x.split(",")[0] for x in data]
    label = [id2label[int(x.split(",")[2])] for x in data]
    label_id = [int(x.split(",")[2]) for x in data]
    start = [float(x.split(",")[3]) for x in data]
    end = [float(x.split(",")[4]) for x in data]
    startFrame = [float(x.split(",")[5]) for x in data]
    endFrame = [float(x.split(",")[6]) for x in data]

    frame_count = video_frame()

    frame_duration = [frame_count[x] for x in video]
    time_duration = [float(x)/30 for x in frame_duration]
    fps = [30 for i in range(len(video))]
    feature_dim = [2048 for i in range(len(video))]
    subset = [splitset for i in range(len(video))]
    
    database = gen_json(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset)
    return database

def gen_view_json(name):

    p1  = './json/{}_Val_Annotation.csv'.format(name)
    p2  = './json/{}_Test_Annotation.csv'.format(name)
    f1 = open(p1,'w',encoding='utf-8',newline='')
    csv_writer1 = csv.writer(f1)
    csv_writer1.writerow(["video","type","type_idx","start","end","startFrame","endFrame"])

    f2 = open(p2,'w',encoding='utf-8',newline='')
    csv_writer2 = csv.writer(f2)
    csv_writer2.writerow(["video","type","type_idx","start","end","startFrame","endFrame"])




    # path_to_file = os.path.join("/mnt/cache/shangjingjie1/AI_city_window/sf_18class_right_150.csv")
    path_to_labelfile = os.path.join("./{}_fix.csv".format(name))


    csv_list_time = []
    with open(path_to_labelfile)as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            csv_list_time.append(row)

    import json

    cover_rate = 0.99

    for i_t, row_t in enumerate(csv_list_time):
        if i_t!=0:
                s1,s2 = 0,0
                label = None
                start_second_t = float(row_t[2])
                end_second_t = float(row_t[3])
                if int(row_t[1]) !=18:
                        # print("start_second_t ",start_second_t)
                        # print("end_second_t ",end_second_t)
                        # print("-----------------------------------")
                        type_idx = int(row_t[1])
                        type_name = "Unkown"
                        if row_t[0]!="Dashboard_User_id_38058_NoAudio_0"and row_t[0]!="Dashboard_User_id_38058_NoAudio_1"and  row_t[0]!="Right_side_window_User_id_38058_NoAudio_0"and  row_t[0]!="Right_side_window_User_id_38058_NoAudio_1"and  row_t[0]!="Rear_view_User_id_38058_NoAudio_1"and  row_t[0]!="Rear_view_User_id_38058_NoAudio_0":
                            csv_writer1.writerow([row_t[0],type_name,type_idx,start_second_t,end_second_t,start_second_t*30,end_second_t*30])
                        else:
                            csv_writer2.writerow([row_t[0],type_name,type_idx,start_second_t,end_second_t,start_second_t*30,end_second_t*30])
            # anno["annotations"] = label_list
            # json_label[window_namer] = anno

    def gen_json(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset):
        database={}
        for(videox,labelx,label_idx,startx,endx,startFramex,endFramex,frame_durationx,time_durationx,fpsx,feature_dimx,subsetx) in zip(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset):
            if videox not in database:
                database[videox]={}
                database[videox]["subset"]=subsetx
                database[videox]["duration"]=time_durationx
                database[videox]["duration_frame"]=frame_durationx
                database[videox]["fps"]=fpsx
                database[videox]["feature_dim"]=feature_dimx
                database[videox]["annotations"]= []
                database[videox]["annotations"].append({"label":labelx,"segment":[startx,endx],"segment(frames)":[startFramex,endFramex],"label_id":label_idx})
            else:
                database[videox]["annotations"].append({"label":labelx,"segment":[startx,endx],"segment(frames)":[startFramex,endFramex],"label_id":label_idx})
        return database

    def turn_data_format(jjcsv_path,splitset="train"):  # view dash rear right

        id2label={0:"Normal Forward Driving",1:"Drinking",2:"Phone Call(right)",3:"Phone Call(left)",4:"Eating",5:"Text (Right)",\
        6:"Text (Left)",7:"Hair / makeup",8:"Reaching behind",9:"Adjust control panel",10:"Pick up from floor (Driver)",\
        11:"Pick up from floor (Passenger)",12:"Talk to passenger at the right",13:"Talk to passenger at backseat",14:"yawning",\
        15:"Hand on head",16:"Singing with music",17:"shaking or dancing with music"}

        f=open(jjcsv_path)
        data = f.readlines()
        data = data[1:]
        f.close()

        video = [x.split(",")[0] for x in data]
        label = [id2label[int(x.split(",")[2])] for x in data]
        label_id = [int(x.split(",")[2]) for x in data]
        start = [float(x.split(",")[3]) for x in data]
        end = [float(x.split(",")[4]) for x in data]
        startFrame = [float(x.split(",")[5]) for x in data]
        endFrame = [float(x.split(",")[6]) for x in data]

        frame_count = video_frame()

        frame_duration = [frame_count[x] for x in video]
        time_duration = [float(x)/30 for x in frame_duration]
        fps = [30 for i in range(len(video))]
        feature_dim = [2048 for i in range(len(video))]
        subset = [splitset for i in range(len(video))]
        
        database = gen_json(video,label,label_id,start,end,startFrame,endFrame,frame_duration,time_duration,fps,feature_dim,subset)
        return database


    f1.close()
    f2.close()
    import json
    jjcsv_path = p1#"jjdata/{name}_Val_Annotation.csv"
    database_train = turn_data_format(jjcsv_path,"train")

    jjcsv_path = p2#"jjdata/{name}_Test_Annotation.csv"
    database_val = turn_data_format(jjcsv_path,"val")
    
    os.remove(p1)
    os.remove(p2)

    database={}
    for k,v in database_train.items():
        database[k] = v

    for k,v in database_val.items():
        database[k] = v

    ret_json={"version": "AICITY-30fps","database":database}

    json.dump(ret_json, open("./json/aicity_{}.json".format(name),"w"))

gen_view_json('rear')
gen_view_json('right')
gen_view_json('dashboard')

def gen_test_feat_anno_file(name):
    json_data = json.load(open("./json/aicity_{}.json".format(name),"rb"))
    frame_count = video_frame(False)

    video=[]
    for x in list(frame_count.keys()):
        if("right" in x.lower()):
            video.append(x)
    
    frame_duration = [frame_count[x] for x in video]
    time_duration = [float(x)/30 for x in frame_duration]
    fps = [30 for i in range(len(video))]
    feature_dim = [2048 for i in range(len(video))]
    subset = ["test" for i in range(len(video))]

    for(videox,frame_durationx,time_durationx,fpsx,feature_dimx,subsetx) in zip(video,frame_duration,time_duration,fps,feature_dim,subset):
        if videox not in json_data["database"]:
            json_data["database"][videox]={}
            json_data["database"][videox]["subset"]=subsetx
            json_data["database"][videox]["duration"]=time_durationx
            json_data["database"][videox]["duration_frame"]=frame_durationx
            json_data["database"][videox]["fps"]=fpsx
            json_data["database"][videox]["feature_dim"]=feature_dimx
    
    json.dump(json_data, open("./json/aicity_{}.json".format(name),"w"))

gen_test_feat_anno_file('rear')
gen_test_feat_anno_file('right')
gen_test_feat_anno_file('dashboard')


