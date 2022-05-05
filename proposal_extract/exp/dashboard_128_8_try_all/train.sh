work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python ./train.py $work_path/config.yaml --output ${work_path}/ckpt \
    2>&1 | tee ${work_path}/log.txt