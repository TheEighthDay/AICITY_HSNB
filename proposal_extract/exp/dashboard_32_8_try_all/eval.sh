work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p GVT -n1 --gres=gpu:2 --ntasks-per-node=1 --cpus-per-task=10 --comment=spring-submit --job-name test \
python ./eval.py $work_path/config.yaml $work_path/ckpt --saveonly \
    2>&1 | tee -a ${work_path}/log.txt