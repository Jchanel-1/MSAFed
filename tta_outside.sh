temperature_contrast=0.07
mkdir -p "${BASE_PATH}/"
comm_round=5
md=test
gpu=6
num_cluster=10
agg_coe=0.1
SRC_BASE_PATH="./exps/iada/${agg_coe}w/nc${num_cluster}"
BASE_PATH="./exps/tta/tt1.0/"
seed=1234
src_base_path="${SRC_BASE_PATH}/1/seed${seed}/nc${num_cluster}"
base_path="${BASE_PATH}/nc$num_cluster/1/seed${seed}/nc${num_cluster}"
mkdir -p "$base_path"
    python ./main.py \
    --data polyp --source 2 3 4 --target 1  \
    --tensor_log $base_path --lr_path "${src_base_path}/adalr.pt" \
    --wei_path "${src_base_path}/all_ada.pt" \
    --comm_round ${comm_round} \
    --save_prototype_path $src_base_path \
    --gpu $gpu --ood_test --md $md \
    --base_path ${base_path} >> "${base_path}/log.txt" 2>&1 &
