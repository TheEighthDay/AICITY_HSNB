import os


def get_file_dict():
    file_dict = {}
    for p1 in os.listdir(path):
        p2 = os.path.join(path, p1)
        if os.path.isdir(p2):
            for p3 in os.listdir(p2):
                final_path = os.path.join(p2, p3)
                cur_path = os.path.join(p1, p3)
                file_dict[cur_path] = len(os.listdir(final_path))
    return file_dict

if __name__ == '__main__':
    frame = 128
    stride = 32
    onlyview = 'dashboard'
    path = '/mnt/lustre/share_data/shangjingjie1/AiCityClip/A1_frame'
    prefix = './snippet_data/snippet_dashboard_frame128_stride32_A1'

    if not os.path.exists(prefix):
        os.makedirs(prefix)


    # path = '/mnt/lustre/share_data/shangjingjie1/AiCityClip/A2'
    # prefix = './snippet_data/snippet_dashboard_frame128_stride32_A2'
    file_tmpl = 'all_{}.csv'


    view_list = [onlyview]
    file_list = []
    for v in view_list:
        file_list.append(open(os.path.join(prefix, file_tmpl.format(v)), 'w'))

    file_dict = get_file_dict()

    for path, length in file_dict.items():
        print(path, length)
        for i, view in enumerate(view_list):
            if view in path.lower():
                for j in range(0, length-frame, stride):
                    start = str(j)
                    end = str(j + frame)
                    file_list[i].writelines(','.join([path, start, end]) + '\n')
