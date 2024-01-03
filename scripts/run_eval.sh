
MODEL=$1
INPUT_NAME=$2
DATASET=tiagoblima/qg_squad_v1_pt

python src/eval.py -m $MODEL -i $INPUT_NAME -d $DATASET --metrics "sacrebleu,rouge,meteor,ter"