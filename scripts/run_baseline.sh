export $(grep -v '^#' /content/question-generation-ptbr/scripts/.env | xargs)
WANDB_PROJECT=question-generation-ptbr

python /content/question-generation-ptbr/src/run_seq2seq_qg.py \
  --model_name_or_path unicamp-dl/ptt5-small-t5-vocab \
  --dataset_name tiagoblima/qg_squad_v1_pt \
  --context_column paragraph \
  --question_column question \
  --answer_column answer \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 64 \
  --learning_rate 0.0001 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --push_to_hub \
  --report_to "wandb" \
  --run_name "exp-baseline" \
  --push_to_hub_token $HUGGINGFACE_TOKEN \
  --output_dir /tmp/debug_t5-small_squad/ \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --evaluation_strategy "steps"