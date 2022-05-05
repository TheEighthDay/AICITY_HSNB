work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../slowfast \
python tools/run_net_ema.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR /mnt/lustre/tiankaibin/aicity/data \
  DATA.PATH_PREFIX /mnt/lustre/share_data/shangjingjie1/AiCityClip/A1 \
  DATA.LABEL_PATH_TEMPLATE "{}_split{}_without18_rearview.csv" \
  DATA.IMAGE_TEMPLATE "{:05d}.jpg" \
  DATA.SPLIT 999 \
  DATA.PATH_LABEL_SEPARATOR "," \
  TEST.TEST_BEST True \
  DATA.MC True \
  RNG_SEED 6666 \
  DATA.TEST_CROP_SIZE 256 \
  TEST.ENABLE True \
  TRAIN.ENABLE False \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 3 \
  NUM_GPUS 4 \
  OUTPUT_DIR $work_path
