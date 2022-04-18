

# Train classification model.
## 1.Data collection
The category data list csv file should be preprocessed as the following format in `pre-processing/generate_cla_csv/README.md`:
```
user_id_49381/Dashboard_user_id_49381_NoAudio_0_1,480,18,18,3,49381,Dashboard,None
```
## 2.Train model
We decord the video as frames in `pre-processing/gen_frame/README.md`.
The init slowfast weight SlowFast-ResNet101-8x8 can be downloaded in  [link](https://drive.google.com/drive/folders/1AY8DAEU3Eepnh4xwS21VLGos43h-7DD3?usp=sharing), 
and we need replace `slowfast/models/sf.py/model_path` as the downloaded folder.

In the `exp` folder, `dashboard_view_32frames` means the model is trained based on dashboard view and sample 32 frames for every samples. 
Train all 9 folder in `exp` for model fusion.
Start the script `run.sh` after updating the followings.

* `DATA.PATH_TO_DATA_DIR` : the folder of category data list csv.
* `DATA.PATH_PREFIX` : the decorded frames folder.
* `DATA.LABEL_PATH_TEMPLATE`: list csv name missing split number.
* `DATA.SPLIT`: split number.

```
bash exp/dashboard_view_32frames/run.sh
```

The checkpoint files will be saved in `dashboard_view_32frames/ema/checkpoints`, we select the last epoch checkpoint for the next steps.

# Inference model to extract features.
We prepare features for the actionformer to get action proposals.

## 1. Data collection
Generate snippet data lists csv through `generate_extract.py`, the followings need to be updated.

* `frame`: the snippet frames.
* `stride`: the snippet stride.
* `onlyview`: the view of snippet.
* `path`:  the decorded frames folder.
* `prefix`: the save folder.

After  `generate_extract.py` the `snippet_data` folder will have snippet data list folders, such as `snippet_dashboard_frame128_stride32_A1`, `snippet_dashboard_frame128_stride32_A2`, and etc. We need to generate \[dashboard view, 32 frames, 8stride\],\[dashboard view, 128 frames, 32stride\],\[dashboard view, 128 frames, 8stride\],\[right view, 128 frames, 32stride\] for bath A1 and A2(B) dataset.

## 2.Inference
According to snippet data lists, we get the snippet fatures.

In the `exp_extract_features` folder, `dashboard_A1_128_8` means the exp folder to extract dashboard view snippet features with 128 frames and stride 8 for A1. We need to extract the snippet features for all the snippet data lists. The following parameters should update before running script.

* `DATA.PATH_TO_DATA_DIR`: snippet data list csv folder.
* `DATA.PATH_PREFIX`: the decorded frames folder.
* `DATA.LABEL_PATH_TEMPLATE`: snippet data list csv name.
* `TEST.CHECKPOINT_FILE_PATH`: the saved model checkpoint path.


```
bash exp_extract_features/dashboard_A1_128_8/run.sh
```


Note that, the `dashboard_A1_128_8` should use the last ema checkpoint in `exp/dashboard_view_128frames`, as the checkpoint is trained from 128 frames sampling. 

After inferencing, the pkl files like `fea_128x256x1x3.pkl` will be saved in exp folder. We need process this files to generate features through `deal_pkl.py`. The following parameters should update before running script.

* `extract_path`:  snippet data list csv path.
* `prefix`: the exp folder of generate snippet.
* `save_path`: save folder of snippet features.
* `pkl_path`: the pkl name in the exp folder.

Note that, we save the snippet features in `proposal_extract/data/dashboard_view_128_8_features_A1` corresponding to the pkl in `dashboard_A1_128_8`.

Finally, there are four sinppet features folder in `proposal_extract/data/`.

# Inference model to predict proposals.

Before predicting proposals, we have generated proposals following  "proposal_extract/README.md".

For every `ckpt/eval.csv` proposal file, we use 9 classfication model mentioned in section `Train model`, to  predict the category for each proposal.

There are 9 exp folder in `exp_predict_proposals` corresponding to 9 exp folder in `exp`, in each exp folder, we need to update the followings:

run.sh
* `DATA.LABEL_PATH_TEMPLATE` : proposal file path
* `TEST.CHECKPOINT_FILE_PATH` : the weight path of corresponding classification model.
* `DATA.PATH_PREFIX`: the decorded frames folder.

config.yaml
* `DATA.NUM_FRAMES`: the number of frames is equal to the number of segment sample frames in corresponding classification model.

Conduct the script to start predict.
```
bash exp_predict_proposals/dashboard_view_32frames_predict/run.sh
```

After predicting, the category label pkl will saved in every exp folder, such as `fea_32x256x1x3.pkl`. 
We can collect 9 category label pkl files for one proposal file.
We totally collect four groups category label pkl files: 4x9 into four folder.

We will use all 4 proposal files and 36 category label pkl files in the post-processing phase, See `post-processing/README.md` for how to organize and post-process them.






