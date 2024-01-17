MODEL=$1
INPUT_NAME=$2
DATASET=tiagoblima/qg_br_squadv2
OUTPUT_DIR="/content/drive/MyDrive/QuestionGeneration/Reports/"
python src/eval.py -m $MODEL \
                   -i $INPUT_NAME  \
                   -d $DATASET \
                   -o $OUTPUT_DIR \
                   --metrics "sacrebleu,rouge,meteor,ter,bertscore"