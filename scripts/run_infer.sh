MODEL=$1
INPUTS=$2
DATASET=tiagoblima/preprocessed-du-qg-squadv1_pt
OUTPUT_DIR="/content/drive/MyDrive/QuestionGeneration/Reports/"
python src/infer.py -m $MODEL  \
                   -d $DATASET \
                   -i $INPUTS \
                   -o $OUTPUT_DIR