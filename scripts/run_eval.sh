MODEL=$1
INPUT_NAME=$2
DATASET=tiagoblima/qg_faquad
OUTPUT_DIR="/content/drive/MyDrive/QuestionGeneration/Reports/$1"
hyp_test=""
python src/eval.py -m $MODEL \
                   -i $INPUT_NAME  \
                   --hyp_test $hyp_test \
                   -d $DATASET \
                   -o $OUTPUT_DIR
                  