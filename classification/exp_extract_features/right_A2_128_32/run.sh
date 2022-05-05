work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../slowfast \
python tools/run_extract.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR /mnt/lustre/tiankaibin/aicity/actionformer_release_aicity/data/extract_right_split_A2_128_stride32 \
  DATA.PATH_PREFIX /mnt/lustre/share_data/shangjingjie1/AiCityClip/A2 \
  DATA.LABEL_PATH_TEMPLATE "all_right.csv" \
  DATA.IMAGE_TEMPLATE "{:05d}.jpg" \
  DATA.PATH_LABEL_SEPARATOR "," \
  TEST.CHECKPOINT_FILE_PATH /mnt/lustre/tiankaibin/aicity/uniformer_competition/exp/sf_32_vitaug_no18_lr5e-5e200_do0_sparse_new_rightsidewindow_128/ema/checkpoints/checkpoint_epoch_00200.pyth \
  DATA.MC True \
  DATA.TEST_CROP_SIZE 256 \
  TEST.BATCH_SIZE 32 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 3 \
  NUM_GPUS 8 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path
