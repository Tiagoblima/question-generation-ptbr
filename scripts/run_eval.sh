MODEL=$1
PRED_FILE=$2
DATASET=tiagoblima/qg_squad_v1_pt
OUTPUT_DIR="/content/drive/MyDrive/QuestionGeneration/Reports/$1"
python src/eval.py --pred_file $PRED_FILE  \
                   -d $DATASET \
                   -o $OUTPUT_DIR