
MODEL=$1
DATASET=$2
METRICS=$3

python src/eval.py -m $MODEL  -d $DATASET --metrics $METRICS