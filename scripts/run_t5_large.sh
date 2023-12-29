SIZE=large
BASE_MODEL=unicamp-dl/ptt5-${SIZE}-t5-vocab


python src/run_seq2seq_qg.py \
  --model_name_or_path $BASE_MODEL \
  --dataset_name tiagoblima/qg_squad_v1_pt \
  --context_column paragraph \
  --question_column question \
  --answer_column answer \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 1 \
  --learning_rate 0.0001 \
  --gradient_checkpointing \
  --gradient_accumulation 8 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --save_total_limit 3 \
  --push_to_hub \
  --report_to "wandb" \
  --run_name "t5_$SIZE" \
  --push_to_hub_token $HUGGINGFACE_TOKEN \
  --output_dir /tmp/debug_t5-${SIZE}_squad/