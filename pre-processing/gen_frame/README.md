**## Extracting frames**



Before training and testing, We should extract frames from A1 and A2 and generate the new csv file for training.


First, you should modify the path arguements in generate_frame_and_new_csv.sh.
For classification training frames extracting(24026.py 24491.py 35133.py 38058.py 49381.py)
\```

    --path_to_csv_and_origin_video  your_path_to_training_video_and_csv (e.g."./user_id_24026/user_id_24026.csv") \
    --path_to_save_frame your_path_to_save_frames (e.g. "./user_id_24026_frame")\
    --path_to_save_new_csv your_path_to_save_new_csv(e.g. "./user_id_24026_clip.csv")


\```
For proposal training frames extracting(A1.py A2.py)
\```

    --path_to_save_frame your_path_to_save_frames(e.g. "./A1_frame/") \
    --path_to_origin_video your_path_to_save_new_csv(e.g. "./AiCityData/A1/")


\```

Then, try this command.


\```

sh ./generate_frame_and_new_csv.sh

\```

