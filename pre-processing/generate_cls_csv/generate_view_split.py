import os

split_type = ['train', 'test']

label_file = './{}_split1.csv'
# without18_tpl = './{}_split1_without18.csv'
without18_view_tpl = './{}_split1_without18_{}.csv'


view_list = ['Dashboard', 'Rightside_window', 'Rear_view']
user_list = ['24491', '24026', '35133', '38508', '49381']


for st in split_type:
    # cur_file = open(without18_tpl.format(st), 'w')
    view_file = list()
    for view in view_list:
        name = ''.join(view.lower().split('_'))
        view_file.append(open(without18_view_tpl.format(st, name), 'w'))
    with open(label_file.format(st), 'r') as f:
        lines = f.readlines()
        for line in lines:
            msg = line.split(',')
            cls_num = int(msg[2])
            if cls_num != 18:
                # cur_file.writelines(line)
                cur_view = msg[-2]
                for i, view in enumerate(view_list):
                    if cur_view == view:
                        view_file[i].writelines(line)

    # cur_file.close()
    for f in view_file:
        f.close()