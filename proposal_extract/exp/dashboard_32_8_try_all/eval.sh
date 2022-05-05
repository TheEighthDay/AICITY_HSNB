work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python ./eval.py $work_path/config.yaml $work_path/ckpt --saveonly \
    2>&1 | tee -a ${work_path}/log.txt