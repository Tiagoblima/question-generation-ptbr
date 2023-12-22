SIZE="base"
BASE_MODEL=facebook/mbart-large-50


python src/run_seq2seq_qg.py \
  --model_name_or_path $BASE_MODEL \
  --dataset_name tiagoblima/qg_squad_v1_pt \
  --context_column paragraph \
  --question_column question \
  --answer_column answer \
  --do_train \
  --do_eval \
  --src_lang "pt_XX" \
  --tgt_lang "pt_XX" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation 2 \
  --learning_rate 0.0001 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_total_limit 1 \
  --push_to_hub \
  --use_fast_tokenizer \
  --report_to "wandb" \
  --run_name "mbart-large" \
  --push_to_hub_token $HUGGINGFACE_TOKEN \
  --output_dir /tmp/debug_t5-${SIZE}_squad/
