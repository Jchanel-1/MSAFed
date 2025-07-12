num_cluster=10
BASE_PATH="./exps/iada"
gpu=6
seed=1234
mkdir -p "${BASE_PATH}/"
    SRC_BASE_PATH="./exps/pretrain/0.1w/nc${num_cluster}"
    src_base_path="${SRC_BASE_PATH}/all/seed${seed}"
    base_path="${BASE_PATH}/0.1w/nc${num_cluster}/all/seed${seed}/nc${num_cluster}"
    mkdir -p "$base_path"
    /home/jjj1/miniconda3/envs/fedl/bin/python ./main.py \
    --data polyp --source 1 2 3 4 --train \
    --tensor_log $base_path --is_vis_adalr --save_lr --save_lr_path "${base_path}/adalr.pt" \
    --save_weights --save_weights_path "${base_path}/all_atp.pt" \
    --pt_path "${src_base_path}/all.pt" --comm_round 10 --seed $seed \
    --save_prototype --save_prototype_path $base_path \
    --num_cluster $num_cluster \
    --gpu $gpu >> "${base_path}/log.txt" 2>&1 &


