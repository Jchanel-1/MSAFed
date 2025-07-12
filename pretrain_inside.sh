temperature_contrast=0.07
BASE_PATH="./exps/pretrain"
mkdir -p "${BASE_PATH}/"
gpu=6
comm_round=200
num_cluster=10
seed=1234

base_path="$BASE_PATH/0.1w/nc$num_cluster/all/seed${seed}"
mkdir -p "${base_path}/"
echo "$base_path"
/home/jjj1/miniconda3/envs/fedl/bin/python ./main.py \
--data polyp --source 1 2 3 4 --pretrain \
--base_path $base_path --tensor_log $base_path \
--gpu $gpu --seed $seed \
--comm_round $comm_round \
--save_global_weights --save_global_path $base_path/all.pt >> "${base_path}/log.txt" 2>&1 &
