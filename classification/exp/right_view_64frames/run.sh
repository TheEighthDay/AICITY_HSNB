work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p GVT -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=40 --comment=spring-submit --job-name sf \
python tools/run_net_ema.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR /mnt/lustre/tiankaibin/aicity/data \
  DATA.PATH_PREFIX /mnt/lustre/share_data/shangjingjie1/AiCityClip/A1 \
  DATA.LABEL_PATH_TEMPLATE "{}_split{}_without18_rightsidewindow.csv" \
  DATA.IMAGE_TEMPLATE "{:05d}.jpg" \
  DATA.SPLIT 999 \
  DATA.PATH_LABEL_SEPARATOR "," \
  TEST.TEST_BEST True \
  DATA.MC True \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path
