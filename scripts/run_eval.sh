BASE_DIR="./reports/tiagoblima/mt5_base-qg-af-oficial"
PRED_FILE="${BASE_DIR}/hypothesis.txt"
REF_FILE="./data/du-squadv1/test.ref"

python src/eval.py --pred_file $PRED_FILE  \
                   --ref_file $REF_FILE \
                   -o $BASE_DIR