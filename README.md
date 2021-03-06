# Code Release for AICITY_2022 Track3 HSNB Team
We make good use of the multi-view synchronization among videos, and conduct robust **Multi-View Practice (MVP)** for driving action localization. To avoid overfitting, we finetune SlowFast with Kinetics-700 pre-training as the feature extractor. Then the features of different views are passed to ActionFormer to generate candidate action proposals. For precisely localizing all the actions, we design elaborate post-processing, including model voting, threshold filtering and duplication removal.
![](data/frame.jpg)
More details can be found in our workshop paper: **MVP: Robust Multi-View Practice for Driving Action Localization**. [Arxiv](https://arxiv.org/pdf/2207.02042.pdf)

## Pre-processing

There are three steps and the details are explained in `README.md` under the corresponding folder:

* [pre-processing/gen_frame](./pre-processing/gen_frame/): In this folder, we decord the original video data as  frames.
* [pre-processing/gen_json_and_fea](./pre-processing/gen_json_and_fea/): In this folder, we prepare the action segment temporal location labels as the json format, and collect meta info for all videos, like duration, duration_frame and etc.
* [pre-processing/generate_cls_csv](./pre-processing/generate_cls_csv/): In this folder, we prepare  the action segment classfication labels as the csv format.


## Method


There are five steps:

1. [classification/README.md](./classification/README.md)(Train model): Train the basic classification model for  action segments.
2. [classification/README.md](./classification/README.md)(Inference model to extract features): Use the well-trained classification model to extract snippet features.
3. [proposal_extract/README.md](./proposal_extract/README.md)(Train and inference model): Use the snippet features to the train temporal location model, and infercence test dataset to generate proposals.
4. [proposal_extract/README.md](./proposal_extract/README.md)(Convert format): Convert proposals from pkl format to csv format.
5. [classification/README.md](./classification/README.md)(Inference model to predict proposals): Classify the generated proposals.


## Post-processing 

Conduct post-processing following [post-processing/README.md](./post-processing/README.md).

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Acknowledgment 

We are very grateful to the organizers for providing this opportunity for us, to explore the model in real multi-view driving videos.
This is very meaningful. We believe it will promote the development of ai city and automatic driving.

In addition, our code is built based on [UniFormer](https://github.com/Sense-X/UniFormer), [SlowFast](https://github.com/facebookresearch/SlowFast) and [ActionFormer](https://github.com/happyharrycn/actionformer_release).
