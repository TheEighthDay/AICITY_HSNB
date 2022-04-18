import os
import pickle


def turn_kcfeature(pkldata,save_dir="sf_features",fmt="fea"): #fea pred
    # pkldata = pickle.load(open(kcfea_path,"rb"))
    #'user_id_79336/Dashboard_User_id_79336_NoAudio_0'
    for video in tqdm(pkldata.keys()):
        if fmt == "pred":
            savevideo = video.split("/")[1]+"_"+"pred.npy"
        else:
            savevideo = video.split("/")[1]+".npy"
        
        np.save(os.path.join(save_dir,savevideo),pkldata[video][fmt+"s"].numpy())


# dashboard
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_dashboard/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_dashboard_A2/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_dashboard_notrain/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class2_dashboard/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class2_dashboard_A2/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_dashboard_dense/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_dashboard_dense_A2/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_dashboard.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_dashboard_notrain_A2/'
# pkl_path = 'fea_32x256x1x3.pkl'
# new_pkl_path = 'dashboard_fea_32x256x1x3.pkl'

# rear
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_rear.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rear/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_rear.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rear_A2/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_rear.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rear_notrain_A2/'
# pkl_path = 'fea_32x256x1x3.pkl'
# new_pkl_path = 'rear_fea_32x256x1x3.pkl'

# rightside
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_right.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rightside/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_right.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rightside_A2/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_right.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rightside_notrain/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split/all_right.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rightside_dense/'
# extract_path = '/mnt/lustre/share_data/likunchang.vendor/AiCityClipV2/extract_split_A2/all_right.csv'
# prefix = '/mnt/lustre/likunchang.vendor/code/uniformer_competition/exp_extract/class18_rightside_dense_A2/'
extract_path = '/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/extract_right_split_A2_128_stride32/all_right.csv'
prefix = '/mnt/lustre/tiankaibin/aicity/uniformer_competition/exp_extract/right_A2_128_32/'
save_path = "xxx/features"
pkl_path = 'fea_128x256x1x3.pkl'


path_dict = {}
last_path = None
last_end_index = None
flag = False
with open(extract_path, 'r') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        msg = line.rstrip().split(',')
        path, start, end = msg
        if int(start) == 0:
            path_dict[path] = [count]
            if last_path:
                path_dict[last_path].append(count - 1)
        last_path = path
        last_end_index = count
        count += 1
    path_dict[last_path].append(count - 1)

print(path_dict)


cur_pkl = pickle.load(open(os.path.join(prefix, pkl_path), 'rb'))
new_dict = {}
for path, start_end in path_dict.items():
    start, end = start_end

    new_dict[path] = {}
    new_dict[path]['feas'] = cur_pkl['video_feas'][start: end+1]
    new_dict[path]['preds'] = cur_pkl['video_preds'][start: end+1]
    new_dict[path]['frame'] = cur_pkl['video_end'][end]

turn_kcfeature(new_dict,save_path)


# with open(os.path.join(save_path, new_pkl_path), 'wb') as f:
#     pickle.dump(new_dict, f)
