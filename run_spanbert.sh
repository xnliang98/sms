python examples/relation_extraction/run_recls.py \
        --data_dir data/tacred/ \
        --model_type spanbert \
        --model_name_or_path data/spanbert-hf \
        --task_name tacred \
        --output_dir data/output_spanbert_large/ \
        --do_train \
        --do_eval \
        --do_predict \
        --per_gpu_train_batch_size 64 \
        --per_gpu_eval_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 5.0 \
        --overwrite_output_dir \
        --logging_steps 500 \
        --save_steps 500 \
        --warmup_steps 1000 \
        --learning_rate 3e-5  \
        --eval_all_checkpoints



