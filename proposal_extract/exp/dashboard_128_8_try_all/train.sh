work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p GVT -n1 --gres=gpu:2 --ntasks-per-node=1 --cpus-per-task=10 --comment=spring-submit --job-name ac \
python ./train.py $work_path/config.yaml --output ${work_path}/ckpt \
    2>&1 | tee ${work_path}/log.txt