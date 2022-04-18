import torch
import pickle
import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gen_label_list_path', type=str, default="/mnt/lustre/tiankaibin/ug/data/train_unlabel.csv")
parser.add_argument('--gen_label_pkl_path', type=str, default="exp/uniformer_s16x4_k600_dp0.2_e20_1e-4_unlabeled_labeled_data/probs/second_dark_train_unlabel_16x224x1x1.pkl")
args = parser.parse_args()


pred = pickle.load(open(args.gen_label_pkl_path, 'rb'))
pred_score = torch.Tensor(pred['video_preds']).softmax(-1)
pred_label = pred_score.argmax(-1).squeeze().tolist()

f=open(args.gen_label_list_path,"r")
idx = f.readlines()
f.close()
idx = [x.strip().split(",")[0] for x in idx]

pd.DataFrame({"id":idx,"label":pred_label}).to_csv("pseudo.csv",index=False,header=False)
