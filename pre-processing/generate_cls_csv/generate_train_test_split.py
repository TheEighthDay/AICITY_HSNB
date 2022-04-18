import os

label_file = './total_clip.csv'
split_train_tpl = './train_split{}.csv'
split_test_tpl = './test_split{}.csv'

view_list = ['Dashboard', 'Rightside_window', 'Rear_view']
user_list = ['49381', '38508', '35133', '24491', '24026']


train_file_list = []
test_file_list = []
for i in range(len(user_list)):
    train_file_list.append(open(split_train_tpl.format(i+1), 'w'))
    test_file_list.append(open(split_test_tpl.format(i+1), 'w'))

view_set = set()
len_f_set = set()
with open(label_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        msg = line.split(',')
        user_id = msg[-3]
        if int(msg[2]) == 18:
            len_f_set.add(int(msg[1]))
        view_set.add(msg[-2])
        for i, user in enumerate(user_list):
            if user != user_id:
                train_file_list[i].writelines(line)
            else:
                test_file_list[i].writelines(line)

print(view_set)
print(len_f_set)