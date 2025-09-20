#train
python3 main.py --config /configs/Template-BBDM.yaml --train --sample_at_start --save_top --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

#test
python3 main.py --config /configs/Template-BBDM.yaml --sample_to_eval --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt
