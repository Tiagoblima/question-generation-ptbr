MODEL=$1
PRED_FILE=$2
DATASET=tiagoblima/qg_squad_v1_pt
OUTPUT_DIR="/content/drive/MyDrive/QuestionGeneration/Reports/"
python src/v1/eval.py --pred_file $PRED_FILE  \
                   -d $DATASET \
                   -o $OUTPUT_DIR