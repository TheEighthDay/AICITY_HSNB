# Generate Action Proposals

## Data Collection

We should prepare two kinds of data for the next training step: snippet features and labels. Snippet features are generated following "classfication/README.md":`Inference model to extract features`. The labels are saved in json files, which are generated from  `pre-processing/gen_json_and_fea/README.md`.


## Train and inference model

In the `exp` folder, `dashboard_32_8_try_all` means we use the snippet features in `data/dashboard_view_32_8_features_A1` and the labels in `data/annotations/dashboard.json` to train a actionformer model. To get the proposals of A2(B), we utilze the snippet features  from `data/dashboard_view_32_8_features_A2` to infernece the model.

Update the followings and start training stage:

config.yaml
* dataset.json_file : the label json path
* dataset.feat_folder:  the snippet features folder
* dataset.feat_stride: the snippet features' stride
* dataset.num_frames: the snippet features' frames

```
bash exp/dashboard_32_8_try_all/train.sh
```

After training snippet feature sequences, such as in `dashboard_32_8_try_all`, a new folder named `ckpt` will generate. Model weight will be saved in `ckpt` and `epoch_049.pth.tar` will be used to inference.

Update the followings and start inference stage:

config.yaml
* `dataset.feat_folder`: update the snippet features folder from A1 to A2(B)
```
bash exp/dashboard_32_8_try_all/eval.sh
```
After inferenceing, a pkl file named `eval_results.pkl` will saved in `ckpt` folder, which includes all proposals of A2(B) data.

## Convert format
We need to use `generate_prposal_csv.py` convert proposal pkl files format for  classification model. The followings need to be update.

* `prefix` : the prefix path of  proposal pkl file, which also means the 

After converting format, a file named `eval.csv` will saved in `ckpt` folder. A total of 4 `eval.csv` in `dashboard_32_8_try_all/ckpt`,`dashboard_128_8_try_all/ckpt`,`dashboard_128_32_try_all/ckpt` and `right_128_32_try_all/ckpt` will be generated for subsequent classification.

