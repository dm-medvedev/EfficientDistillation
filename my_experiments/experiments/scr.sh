# !/bin/bash

# # gm-gtn: bsz, learner, outer
# dir='gm-gtn/exp3'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/bsz_100 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_1000 1000

# python3 look_for_results.py $dir/res.json $dir/ls_5 1000
# python3 look_for_results.py $dir/res.json $dir/ls_50 1000

# python3 look_for_results.py $dir/res.json $dir/ous_5 1000
# python3 look_for_results.py $dir/res.json $dir/ous_50 1000

# # gm-gtn: input_count
# dir='gm-gtn/exp4'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/ic_5  200
# python3 look_for_results.py $dir/res.json $dir/ic_10 100

# # gm-gtn: random
# dir='gm-gtn/exp5'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/bsz_100 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_1000 1000

# # dd vs gtn
# dir='dd-vs-gtn/dd'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/ipc_10 --is_dd 1000
# python3 look_for_results.py $dir/res.json $dir/ipc_50 --is_dd 1000
# python3 look_for_results.py $dir/res.json $dir/ipc_100 --is_dd 1000


# # ift-gtn
# dir='ift-gtn/exp2'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/bsz_100 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_300 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500 1000

# python3 look_for_results.py $dir/res.json $dir/mx_1 1000
# python3 look_for_results.py $dir/res.json $dir/mx_50 1000

# python3 look_for_results.py $dir/res.json $dir/trunc_5 1000
# python3 look_for_results.py $dir/res.json $dir/trunc_30 1000

# # ift-gtn: random
# dir='ift-gtn/exp3'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/bsz_100 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_300 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_450 1000

# ift-dd
# dir='ift-dd/exp2'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/ipc_10 --is_dd 1000
# python3 look_for_results.py $dir/res.json $dir/ipc_50 --is_dd 1000
# python3 look_for_results.py $dir/res.json $dir/ipc_60 --is_dd 1000

# python3 look_for_results.py $dir/res.json $dir/mx_1 --is_dd 1000
# python3 look_for_results.py $dir/res.json $dir/mx_50 --is_dd 1000

# python3 look_for_results.py $dir/res.json $dir/trunc_5 --is_dd 1000
# python3 look_for_results.py $dir/res.json $dir/trunc_30 --is_dd 1000

# # dd vs gtn
# dir='dd-vs-gtn/gtn'
# python3 $dir/launch.py

# python3 look_for_results.py $dir/res.json $dir/bsz_100_k16 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500_k16 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_1000_k16 1000

# python3 look_for_results.py $dir/res.json $dir/bsz_100_k32 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500_k32 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_1000_k32 1000

# python3 look_for_results.py $dir/res.json $dir/bsz_100_k64 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500_k64 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_1000_k64 1000

# python3 look_for_results.py $dir/res.json $dir/bsz_100_k128 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_500_k128 1000
# python3 look_for_results.py $dir/res.json $dir/bsz_1000_k128 1000


# # test time augmentation
# dir='aug'
# python3 look_for_results.py $dir/gm-dd-res.json dd-vs-gtn/dd/ipc_10 --is_dd --do_aug 1_000
# python3 look_for_results.py $dir/ift-gtn-random.json ift-gtn/exp3/bsz_100 --do_aug 1_000
# python3 look_for_results.py $dir/ift-dd-res.json ift-dd/exp2/mx_50 --is_dd --do_aug 1_000
# python3 look_for_results.py $dir/ift-gtn.json ift-gtn/exp2/bsz_100 --do_aug 1_000
# python3 look_for_results.py $dir/gm-gtn-res.json dd-vs-gtn/gtn/bsz_500_k128 --do_aug 1_000


# train time augmentation
# dir='tr_aug/gm-dd'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/out --is_dd 1_000
# python3 look_for_results.py $dir/res_aug.json $dir/out --is_dd --do_aug 1_000

# dir='tr_aug/gm-gtn'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/out 1_000
# python3 look_for_results.py $dir/res_aug.json $dir/out --do_aug 1_000

# dir='tr_aug/ift-dd'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/out --is_dd 1_000
# python3 look_for_results.py $dir/res_aug.json $dir/out --is_dd --do_aug 1_000

# dir='tr_aug/ift-gtn'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/out 1_000
# python3 look_for_results.py $dir/res_aug.json $dir/out --do_aug 1_000

# dir='tr_aug/ift-gtn-rand'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/out 1_000
# python3 look_for_results.py $dir/res_aug.json $dir/out --do_aug 1_000


# # generalization
# dir='generalization'
# python3 generalization.py $dir/gm-gtn-res.json dd-vs-gtn/gtn/bsz_500_k128 --do_aug 1_000
# python3 generalization.py $dir/gm-dd-res.json dd-vs-gtn/dd/ipc_10 --is_dd --do_aug 1_000
# python3 generalization.py $dir/ift-gtn-random.json ift-gtn/exp3/bsz_100 --do_aug 1_000
# python3 generalization.py $dir/ift-dd-res.json ift-dd/exp2/mx_50 --is_dd --do_aug 1_000
# python3 generalization.py $dir/ift-gtn.json ift-gtn/exp2/bsz_100 --do_aug 1_000


# # my-gm-gtn
# dir='my-gm-gtn/exp1'
# # python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/rs_2 1_000
# python3 look_for_results.py $dir/res.json $dir/rs_5 1_000
# python3 look_for_results.py $dir/res.json $dir/rs_10 1_000


# # my-gm-dd
# dir='my-gm-dd/exp1'
# # python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir/rs_2 --is_dd 1_000
# python3 look_for_results.py $dir/res.json $dir/rs_5 --is_dd 1_000
# python3 look_for_results.py $dir/res.json $dir/rs_10 --is_dd 1_000


# # Ablation: GM не per class
# dir='ablation/gm-npc-dd'
# # python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir  --is_dd 1_000


# # Ablation: GM не per class
# dir='ablation/gm-npc-gtn'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir 1_000


# Long Train fixed: 50 minutes MNIST
# dir='long_train/gm-gtn-mnist'
# python3 $dir/launch.py
# python3 look_for_results.py $dir/res.json $dir --do_aug  1_000


# Long Train fixed: 50 minutes CIFAR10
dir='long_train/gm-gtn-cifar10'
# python3 $dir/launch.py
python3 look_for_results.py $dir/res.json $dir --do_aug --cifar 1_000
