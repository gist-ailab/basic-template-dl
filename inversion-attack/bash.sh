CUDA_VISIBLE_DEVICES=0 python train_inverter.py --class_cond True --class_type label --data_id 0
CUDA_VISIBLE_DEVICES=5 python train_inverter.py --class_cond True --class_type output --data_id 0
CUDA_VISIBLE_DEVICES=6 python train_inverter.py --class_cond False --class_type none --data_id 0

CUDA_VISIBLE_DEVICES=2 python train_inverter.py --class_cond True --class_type label --data_id 1
CUDA_VISIBLE_DEVICES=2 python train_inverter.py --class_cond True --class_type output --data_id 1
CUDA_VISIBLE_DEVICES=5 python train_inverter.py --class_cond False --class_type none --data_id 1




