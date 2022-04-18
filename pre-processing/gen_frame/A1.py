# from iopath.common.file_io import g_pathmgr
from PIL import Image, ImageOps
import os
import PIL
import decord
from decord import VideoReader
from decord import cpu
from moviepy.editor import VideoFileClip
import csv
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_origin_video', type=str)
parser.add_argument('--path_to_save_frame', type=str)
# parser.add_argument('--path_to_save_new_csv', type=str)

args = parser.parse_args()

#path to save frame
base_path = args.path_to_save_frame#'/mnt/lustre/share_data/shangjingjie1/AiCityClip/A1_frame/'

user_id_24026_video_list = ["Dashboard_User_id_24026_NoAudio_3.mp4","Dashboard_User_id_24026_NoAudio_4.MP4","Rear_view_User_id_24026_NoAudio_3.MP4"
,"Rear_view_User_id_24026_NoAudio_4.MP4","Right_side_window_User_id_24026_NoAudio_3.MP4","Right_side_window_User_id_24026_NoAudio_4.MP4"]
user_id_35133_video_list = ["Dashboard_user_id_35133_NoAudio_0.MP4","Dashboard_user_id_35133_NoAudio_2.MP4","Rear_view_user_id_35133_NoAudio_0.MP4"
,"Rear_view_user_id_35133_NoAudio_2.MP4","Rightside_window_user_id_35133_NoAudio_0.MP4","Rightside_window_user_id_35133_NoAudio_2.MP4"]
user_id_49381_video_list = ["Dashboard_user_id_49381_NoAudio_0.MP4","Dashboard_user_id_49381_NoAudio_1.MP4","Rearview_mirror_user_id_49381_NoAudio_0.MP4"
,"Rearview_mirror_user_id_49381_NoAudio_1.MP4","Right_window_user_id_49381_NoAudio_0.MP4","Right_window_user_id_49381_NoAudio_1.MP4"]
user_id_24491_video_list = ["Dashboard_user_id_24491_NoAudio_0.MP4","Dashboard_user_id_24491_NoAudio_1.MP4","Rear_view_user_id_24491_NoAudio_0.MP4"
,"Rear_view_user_id_24491_NoAudio_1.MP4","Rightside_window_user_id_24491_NoAudio_0.MP4","Rightside_window_user_id_24491_NoAudio_1.mp4"]
user_id_38058_video_list = ["Dashboard_User_id_38058_NoAudio_0.MP4","Dashboard_User_id_38058_NoAudio_1.MP4","Rear_view_User_id_38058_NoAudio_0.MP4"
,"Rear_view_User_id_38058_NoAudio_1.MP4","Right_side_window_User_id_38058_NoAudio_0.MP4","Right_side_window_User_id_38058_NoAudio_1.MP4"]


#path to origin video
b_path = args.path_to_origin_video
path_24026 = b_path +'/user_id_24026'
path_35133 = b_path +'/user_id_35133'
path_49381 = b_path +'/user_id_49381'
path_24491 = b_path +'/user_id_24491'
path_38058 = b_path +'/user_id_38058'

def find_frame(start_min,start_sec,end_min,end_sec):

    start_frame = (start_min*60+start_sec)*30
    end_frame =  (end_min*60+end_sec)*30  

    return start_frame,end_frame

def save_clip(base_path,class_name,video_path,video_name,start_min,start_sec,end_min,end_sec):
    start_frame,end_frame = find_frame(start_min,start_sec,end_min,end_sec)
    floder_name = (video_name.split('.')[0])
    floder_path = os.path.join(base_path,class_name,floder_name)
    if not os.path.exists(floder_path):
        os.makedirs(floder_path)
    path  = video_path + '/' + video_name
    print("start:",video_name)
    vr = VideoReader(path, ctx=cpu(0))
    for seg_ind in range(start_frame,end_frame):
        images = Image.fromarray(vr[seg_ind].asnumpy())
        images = images.resize((int(images.size[0]/(images.size[1]/320)), 320), Image.ANTIALIAS)
        name = floder_path+"/" + f'{int(seg_ind):05d}.jpg'
        images.save(name)
        print(name)




for i in user_id_24026_video_list:
    if i == "Dashboard_User_id_24026_NoAudio_3.mp4" or i == "Right_side_window_User_id_24026_NoAudio_3.MP4"or i == "Rear_view_User_id_24026_NoAudio_3.MP4":
        save_clip(base_path,"user_id_24026",path_24026,i,0,0,8,50)
    elif i == "Dashboard_User_id_24026_NoAudio_4.MP4"or i == "Right_side_window_User_id_24026_NoAudio_4.MP4"or i == "Rear_view_User_id_24026_NoAudio_4.MP4":
        save_clip(base_path,"user_id_24026",path_24026,i,0,0,9,38)

for i in user_id_35133_video_list:
    if i == "Dashboard_user_id_35133_NoAudio_0.mp4" or i == "Rightside_window_user_id_35133_NoAudio_0.MP4"or i == "Rear_view_user_id_35133_NoAudio_0.MP4":
        save_clip(base_path,"user_id_35133",path_35133,i,1,43,9,50)
    elif i == "Dashboard_user_id_35133_NoAudio_2.MP4"or i == "Rightside_window_user_id_35133_NoAudio_2.MP4"or i == "Rear_view_user_id_35133_NoAudio_2.MP4":
        save_clip(base_path,"user_id_35133",path_35133,i,0,41,8,47)

for i in user_id_49381_video_list:
    if i == "Dashboard_user_id_49381_NoAudio_0.mp4" or i == "Right_window_user_id_49381_NoAudio_0.MP4"or i == "Rearview_mirror_user_id_49381_NoAudio_0.MP4":
        save_clip(base_path,"user_id_49381",path_49381,i,0,0,8,46)
    elif i == "Dashboard_user_id_49381_NoAudio_1.MP4"or i == "Right_window_user_id_49381_NoAudio_1.MP4"or i == "Rearview_mirror_user_id_49381_NoAudio_1.MP4":
        save_clip(base_path,"user_id_49381",path_49381,i,0,0,8,55)

for i in user_id_24491_video_list:
    if i == "Dashboard_user_id_24491_NoAudio_0.mp4" or i == "Rightside_window_user_id_24491_NoAudio_0.MP4"or i == "Rear_view_user_id_24491_NoAudio_0.MP4":
        save_clip(base_path,"user_id_24491",path_24491,i,0,0,9,12)
    elif i == "Dashboard_user_id_24491_NoAudio_1.MP4"or i == "Rightside_window_user_id_24491_NoAudio_1.mp4"or i == "Rear_view_user_id_24491_NoAudio_1.MP4":
        save_clip(base_path,"user_id_24491",path_24491,i,0,0,10,50)

for i in user_id_38058_video_list:
    if i == "Dashboard_User_id_38058_NoAudio_0.mp4" or i == "Right_side_window_User_id_38058_NoAudio_0.MP4"or i == "Rear_view_User_id_38058_NoAudio_0.MP4":
        save_clip(base_path,"user_id_38058",path_38058,i,1,10,10,14)
    elif i == "Dashboard_User_id_38058_NoAudio_1.MP4"or i == "Right_side_window_User_id_38058_NoAudio_1.MP4"or i == "Rear_view_User_id_38058_NoAudio_1.MP4":
        save_clip(base_path,"user_id_38058",path_38058,i,1,23,10,41)


14400+1410
