# from iopath.common.file_io import g_pathmgr
from PIL import Image, ImageOps
import os
import PIL
import decord
from decord import VideoReader
from decord import cpu
from moviepy.editor import VideoFileClip
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_csv_and_origin_video', type=str)
parser.add_argument('--path_to_save_frame', type=str)
parser.add_argument('--path_to_save_new_csv', type=str)

args = parser.parse_args()


#path to csv and origin video
path_to_file = os.path.join(args.path_to_csv_and_origin_video)#"/mnt/lustre/share_data/likunchang.vendor/AiCityData/A1/user_id_24026/user_id_24026.csv")
# with g_pathmgr.open(path_to_file, "r") as f:
#     for index, line in enumerate(f.read().splitlines()):#, Camera_View,_,Start_Time,End_Time, Label,Appearance_Block
#         print(type(line),len(line))
video_path = ''
vr = ''
frame_rate = 30
cnt = 0
total_sec = 0
import csv
#path to save frame
base_path = os.path.join(args.path_to_save_frame)#'/mnt/lustre/share_data/shangjingjie1/AiCityClip_test/A1/user_id_24026/'
#path to save new csv
f = open(args.path_to_save_new_csv,'w',encoding='utf-8',newline='')#/mnt/cache/shangjingjie1/AI_City_clip/user_id_24026_clip.csv
csv_writer = csv.writer(f)
csv_writer.writerow(["path", "frame_num", "class_id","last_class_id", "next_class_id", "User_ID", "Camera_View", "Appearance_Block"])
# csv_writer.writerow(["l",'18','男'])
# csv_writer.writerow(["c",'20','男'])
# csv_writer.writerow(["w",'22','女'])

# 5. 关闭文件
# f.close()
def save_clip(flag,base_path,start_frame,frame_rate,second,vr,video_name,seg,class_id,last_class_id,next_class_id,User_ID,Camera_View,Appearance_Block):
    floder_name = (video_name.split('.')[0]+"_{}".format(seg))
    floder_path = os.path.join(base_path,floder_name)
    if not os.path.exists(floder_path):
        os.makedirs(floder_path)
    # print("frame_rate",frame_rate)
    # print("second", second)
    end_frame = start_frame + second * frame_rate
    csv_writer.writerow([floder_path, frame_rate * second,class_id,last_class_id,next_class_id,User_ID,Camera_View,Appearance_Block])
    print("[%s] start clip path:%s second: %d  frame_num: %d start_frame:%d end_frame:%d"% (flag,floder_path,second,frame_rate*second,start_frame,end_frame))
    for i, seg_ind in enumerate(range(start_frame, end_frame)):
        images = Image.fromarray(vr[seg_ind].asnumpy())
        # print("images",images.size,images.size[1],images.size[1])
        images = images.resize((int(images.size[0]/(images.size[1]/320)), 320), Image.ANTIALIAS)
        name = floder_path+"/" + f'{int(i):05d}.jpg'
        images.save(name)
        # print(name)
csv_list = []
with open(path_to_file)as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        csv_list.append(row)
for i, row in enumerate(csv_list):
    if i!=0:
        if row[1]!=''and row[1]!='File Name':
            File_Name = row[1]
            father_path = os.path.abspath(os.path.dirname(path_to_file) + os.path.sep + ".")
            video_name = (row[1][:-2]+"_NoAudio"+row[1][-2:]+".MP4").replace("User","user")
            if video_name == "Rear_view_user_id_49381_NoAudio_0.MP4":
                video_name = "Rearview_mirror_user_id_49381_NoAudio_0.MP4"
            if video_name == "Rear_view_user_id_49381_NoAudio_1.MP4":
                video_name = "Rearview_mirror_user_id_49381_NoAudio_1.MP4"
            if video_name == "Right_side_window_user_id_49381_NoAudio_0.MP4":
                video_name = "Right_window_user_id_49381_NoAudio_0.MP4"
            if video_name == "Right_side_window_user_id_49381_NoAudio_1.MP4":
                video_name = "Right_window_user_id_49381_NoAudio_1.MP4"
            video_path  = os.path.join(father_path,video_name)
            print(video_path)

            clip = VideoFileClip(video_path)
            total_sec = clip.duration
            vr = VideoReader(video_path, ctx=cpu(0))
            cnt = 0
            frame_rate = int(len(vr) / total_sec)
            print("-------------start video:%s  total_frame:%d total_second: %d ------------"%(video_path,len(vr),clip.duration))
            User_ID = row[0]
            # if row[4]!="0:00:00":
            #     # print("------fisrt-------")
            #     print(row[4])
            #     temp = 0
            #     start_min = 0
            #     start_sed = 0
            #     end_min = int(row[4].split(":")[1])
            #     end_sed = int(row[4].split(":")[2])
            #     if start_min != end_min:
            #         temp = 60 * (end_min - start_min)
            #     second = temp + end_sed - start_sed
            #     start_frame = (start_min * 60 + start_sed) * frame_rate
            #     cnt += 1
            #     flag = "first"
            #     class_id = row[6] if row[6] != 'N/A' else 18
            #     last_class_id = None
            #     Camera_View = row[2]
            #     Appearance_Block= row[7]
            #     next_class_id = csv_list[i + 1][6] if csv_list[i + 1][6] != 'N/A' else 18
            #     # if i != len(csv_list) and csv_list[i + 1][4] != row[5] and csv_list[i+1][1]=='':
            #     #     next_class_id = 18
            #     # else:
            #     #     next_class_id = csv_list[i + 1][6]
            #     save_clip(flag,base_path, start_frame, frame_rate, second, vr, video_name,cnt,class_id,last_class_id,next_class_id,User_ID,Camera_View,Appearance_Block)
            #     # print("-------------")
        else:
            pass
        temp = 0
        # print(row)
        # print("------normal-------")
        start_min = int(row[4].split(":")[1])
        start_sed = int(row[4].split(":")[2])
        end_min = int(row[5].split(":")[1])
        end_sed = int(row[5].split(":")[2])+1
        if start_min != end_min:
            temp = 60 * (end_min - start_min)
        # print("start %d:%d" % (start_min, start_sed))
        # print("end %d:%d" % (end_min, end_sed))
        # print("temp", temp)
        second = temp + end_sed - start_sed
        # print("second", second)
        start_frame = (start_min*60 + start_sed)*frame_rate
        cnt += 1
        flag = "normal"
        class_id = row[6] if row[6] != 'N/A' else 18
        # if csv_list[i - 1][5] != row[4]:
        #     last_class_id = 18
        # else:
        #     last_class_id = csv_list[i - 1][6]
        last_class_id = csv_list[i - 1][6] if csv_list[i - 1][6] != 'N/A' else 18
        next_class_id = csv_list[i + 1][6] if csv_list[i + 1][6] != 'N/A' else 18
        Camera_View = row[2]
        Appearance_Block = row[7]
        # if i != len(csv_list) and csv_list[i + 1][4] != row[5] and csv_list[i + 1][1] == '':
        #     next_class_id = 18
        # else:
        #     next_class_id = csv_list[i + 1][6]
        save_clip(flag,base_path, start_frame, frame_rate, second, vr, video_name,cnt,class_id,last_class_id,next_class_id,User_ID,Camera_View,Appearance_Block)
        # print("------normal-------")
        if i != len(csv_list) and csv_list[i + 1][4] != row[5] and csv_list[i+1][1]=='':
            # print("------margin-------")
            temp = 0
            start_min = int(row[5].split(":")[1])
            start_sed = int(row[5].split(":")[2])+1
            end_min = int(csv_list[i + 1][4].split(":")[1])
            end_sed = int(csv_list[i + 1][4].split(":")[2])
            # print("start %d:%d"%(start_min,start_sed))
            # print("end %d:%d" % (end_min, end_sed))
            if start_min != end_min:
                temp = 60 * (end_min - start_min)
            # print("temp",temp)
            second = temp + end_sed - start_sed
            # print("second", second)
            start_frame = (start_min * 60 + start_sed) * frame_rate
            cnt += 1
            flag = "margin"
            class_id = 18
            last_class_id = csv_list[i - 1][6] if csv_list[i - 1][6] != 'N/A' else 18
            next_class_id = csv_list[i + 1][6] if csv_list[i + 1][6] != 'N/A' else 18
            Camera_View = row[2]
            Appearance_Block = row[7]
            if start_min == end_min and end_sed == start_sed:
                print("[margin]:pass start %d:%d end %d:%d"% (start_min, start_sed,end_min, end_sed))
            else:
                save_clip(flag,base_path, start_frame, frame_rate, second, vr, video_name, cnt,class_id,last_class_id,next_class_id,User_ID,Camera_View,Appearance_Block)
            # print("------margin-------")
        # if i!=len(csv_list) and csv_list[i+1][1]!='' and csv_list[i+1][1]!= File_Name:
        #     # print("------last-------")
        #     temp = 0
        #     start_min = int(row[5].split(":")[1])
        #     start_sed = int(row[5].split(":")[2])
        #     end_min = int(total_sec / 60)
        #     end_sed = int(total_sec % 60)
        #     if start_min != end_min:
        #         temp = 60 * (end_min - start_min)
        #     second = temp + end_sed - start_sed
        #     # print("second", second)
        #     start_frame = (start_min * 60 + start_sed) * frame_rate
        #     cnt += 1
        #     flag = "last"
        #     class_id = row[6] if row[6] != 'N/A' else 18
        #     next_class_id = None
        #     last_class_id = csv_list[i - 1][6] if csv_list[i - 1][6] != 'N/A' else 18
        #     Camera_View = row[2]
        #     Appearance_Block = row[7]
        #     # if csv_list[i - 1][5] != row[4]:
        #     #     last_class_id = 18
        #     # else:
        #     #     last_class_id = csv_list[i + 1][6]
        #     save_clip(flag,base_path, start_frame, frame_rate, second, vr, video_name,cnt,class_id,last_class_id,next_class_id,User_ID,Camera_View,Appearance_Block)
            # print("------last-------")
