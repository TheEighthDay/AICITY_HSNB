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
base_path = args.path_to_save_frame#'/mnt/lustre/share_data/shangjingjie1/AiCityClip/A2/'

user_id_42271_video_list = ["Dashboard_user_id_42271_NoAudio_3.MP4","Rear_view_user_id_42271_NoAudio_3.MP4","Right_side_window_user_id_42271_NoAudio_3.MP4"
,"Dashboard_user_id_42271_NoAudio_4.MP4","Rear_view_user_id_42271_NoAudio_4.MP4","Right_side_window_user_id_42271_NoAudio_4.MP4"]
user_id_56306_video_list = ["Dashboard_user_id_56306_NoAudio_2.MP4","Rear_view_user_id_56306_NoAudio_2.MP4","Rightside_window_user_id_56306_NoAudio_2.MP4"
,"Dashboard_user_id_56306_NoAudio_3.MP4","Rear_view_user_id_56306_NoAudio_3.MP4","Rightside_window_user_id_56306_NoAudio_3.MP4"]
user_id_65818_video_list = ["Dashboard_User_id_65818_NoAudio_1.MP4","Rear_view_User_id_65818_NoAudio_1.MP4","Rightside_window_User_id_65818_NoAudio_1.MP4"
,"Dashboard_User_id_65818_NoAudio_2.MP4","Rear_view_User_id_65818_NoAudio_2.MP4","Rightside_window_User_id_65818_NoAudio_2.MP4"]
user_id_72519_video_list = ["Dashboard_User_id_72519_NoAudio_2.MP4","Rearview_mirror_User_id_72519_NoAudio_2.MP4","Right_window_User_id_72519_NoAudio_2.MP4"
,"Dashboard_User_id_72519_NoAudio_3.MP4","Rearview_mirror_User_id_72519_NoAudio_3.MP4","Right_window_User_id_72519_NoAudio_3.MP4"]
user_id_79336_video_list = ["Dashboard_User_id_79336_NoAudio_0.MP4","Rear_view_User_id_79336_NoAudio_0.MP4","Rightside_window_User_id_79336_NoAudio_0.MP4"
,"Dashboard_User_id_79336_NoAudio_2.MP4","Rear_view_User_id_79336_NoAudio_2.MP4","Rightside_window_User_id_79336_NoAudio_2.MP4"]


#path to origin video
b_path = args.path_to_origin_video
path_42271 = os.path.join(b_path,'user_id_42271')
path_56306 = os.path.join(b_path,'user_id_56306')
path_65818 = os.path.join(b_path,'user_id_65818')
path_72519 = os.path.join(b_path,'user_id_72519')
path_79336 = os.path.join(b_path,'user_id_79336')#'/mnt/lustre/share_data/shangjingjie1/AiCityData/A2/

def save_clip(base_path,class_name,video_path,video_name):
    
    floder_name = (video_name.split('.')[0])
    floder_path = os.path.join(base_path,class_name,floder_name)
    if not os.path.exists(floder_path):
        os.makedirs(floder_path)
    path  = video_path + '/' + video_name
    print("start:",video_name)
    vr = VideoReader(path, ctx=cpu(0))
    for seg_ind in range(len(vr)):
        images = Image.fromarray(vr[seg_ind].asnumpy())
        images = images.resize((int(images.size[0]/(images.size[1]/320)), 320), Image.ANTIALIAS)
        name = floder_path+"/" + f'{int(seg_ind):05d}.jpg'
        images.save(name)
        print(name)

for i in user_id_42271_video_list:
    save_clip(base_path,"user_id_42271",path_42271,i)

for i in user_id_56306_video_list:
    save_clip(base_path,"user_id_56306",path_56306,i)

for i in user_id_65818_video_list:
    save_clip(base_path,"user_id_65818",path_65818,i)

for i in user_id_72519_video_list:
    save_clip(base_path,"user_id_72519",path_72519,i)

for i in user_id_79336_video_list:
    save_clip(base_path,"user_id_79336",path_79336,i)
