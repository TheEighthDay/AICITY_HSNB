import os
import csv

prefix = '/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/exp/right_128_32_try_all/ckpt'

# idx2maxframes={"user_id_42271/Dashboard_user_id_42271_NoAudio_3":15989,"user_id_42271/Dashboard_user_id_42271_NoAudio_4":16799,\
# "user_id_56306/Dashboard_user_id_56306_NoAudio_2":16713,"user_id_56306/Dashboard_user_id_56306_NoAudio_3":16621,\
# "user_id_65818/Dashboard_User_id_65818_NoAudio_1":16949,"user_id_65818/Dashboard_User_id_65818_NoAudio_2":16619,\
# "user_id_72519/Dashboard_User_id_72519_NoAudio_2":19739,"user_id_72519/Dashboard_User_id_72519_NoAudio_3":18479,\
# "user_id_79336/Dashboard_User_id_79336_NoAudio_0":18599,"user_id_79336/Dashboard_User_id_79336_NoAudio_2":16349}



# idx2maxframes_update={}

# for k,v in idx2maxframes.items():
#     idx2maxframes_update[k.split("/")[1]] = v



def gen_prposal_csv():
    with open(os.path.join(prefix, 'eval_results.pkl')) as f:
        lines = f.readlines()
    out = open(os.path.join(prefix, 'eval.csv'), "w", newline = "")
    csv_writer = csv.writer(out, dialect = "excel")
    for line in lines:
        name, start, end, _, _ = line.split(' ')
        start_frame = int(float(start) * 30)
        frame = int((float(end) - float(start)) * 30 + 1)
        user_id = name.split('_')[-3]
        path = f'user_id_{user_id}/{name}'
        csv_writer.writerow([path, frame, start_frame])

# def gen_extend_prposal_csv():
#     with open(os.path.join(prefix, 'eval_results.pkl')) as f:
#         lines = f.readlines()
#     out = open(os.path.join(prefix, 'eval_extend.csv'), "w", newline = "")
#     csv_writer = csv.writer(out, dialect = "excel")
#     for line in lines:
#         name, start, end, _, _ = line.split(' ')
#         start_frame = max(int(float(start) * 30)-150,0)

#         frame = int((float(end) - float(start)) * 30 + 1)
#         frame = min(idx2maxframes_update[name]-start_frame,frame+300)

#         user_id = name.split('_')[-3]
#         path = f'user_id_{user_id}/{name}'
#         csv_writer.writerow([path, frame, start_frame])

if __name__=="__main__":
    gen_prposal_csv()
    # gen_extend_prposal_csv()

