import os
import csv

prefix = '/mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/exp/dashboard_try_all/ckpt'


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
