INPUTS=$1
PRED_FILE=$2
DATASET=tiagoblima/preprocessed-du-qg-squadv1_pt
OUTPUT_DIR="/content/drive/MyDrive/QuestionGeneration/Reports/"
python src/eval.py --pred_file $PRED_FILE  \
                   -d $DATASET \
                   -i $INPUTS \
                   -o $OUTPUT_DIR