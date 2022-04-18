## Classfication Dataset Prepreration
`total_clip.csv` contains the video clips generated from A1 videos.

### Generate train/test split
> There are 5 user in A1, thus we selec videos of 4 user as A1-train, and the rest as A1-val.

Run the follow command:
```shell
python3 generate_train_test_split.py
```
It will generate 5 different train/test splits:
- `[train/test]_split1.csv`
- `[train/test]_split2.csv`
- `[train/test]_split3.csv`
- `[train/test]_split4.csv`
- `[train/test]_split5.csv`
Note that we **ONLY use split1** in our experiments.

### Generte different views
> There are 3 views in A1, we train different views individually.

Run the follow command:
```shell
python3 generate_view_split.py
```
It will generate 3 different views for split1:
- `[train/test]_split1_without18_dashboard.csv`
- `[train/test]_split1_without18_rearview.csv`
- `[train/test]_split1_without18_rightsidewindow.csv`

### Merge different views
> For final models, we train all the video of different views.

Run the follow command:
```shell
cat train_split1_without18_dashboard.csv test_split1_without18_dashboard.csv >> total_split1_without18_dashboard.csv
mv total_split1_without18_dashboard.csv train_split1_without18_dashboard.csv
cat train_split1_without18_rearview.csv test_split1_without18_rearview.csv >> total_split1_without18_rearview.csv
mv total_split1_without18_rearview.csv train_split1_without18_rearview.csv
cat train_split1_without18_rightsidewindow.csv test_split1_without18_rightsidewindow.csv >> total_split1_without18_rightsidewindow.csv
mv total_split1_without18_rightsidewindow.csv train_split1_without18_rightsidewindow.csv
```

It will merge train and val videos:
- `train_split1_without18_dashboard.csv`
- `train_split1_without18_rearview.csv`
- `train_split1_without18_rightsidewindow.csv`