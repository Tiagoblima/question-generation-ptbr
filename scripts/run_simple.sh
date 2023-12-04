#HUGGINGFACE_TOKEN=

python /content/question-generation-ptbr/run_seq2seq_qg.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --push_to_hub \
  --push_to_hub_token $HUGGINGFACE_TOKEN \
  --output_dir /tmp/debug_seq2seq_squad/