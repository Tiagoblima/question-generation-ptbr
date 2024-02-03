BASE_DIR="./reports/"
PRED_FILE="${BASE_DIR}/hypothesis.txt"
REF_FILE="./data/du-squadv1/test.ref"
rm -rf metrics.csv
for model_dir in $(ls "$BASE_DIR/tiagoblima/" ); do
    echo "Evaluating $model_dir"
    python src/eval.py --pred_file "$BASE_DIR/tiagoblima/$model_dir/hypothesis.txt"  \
                    --ref_file $REF_FILE \
                    -o $BASE_DIR/results/$model_dir >> metrics.csv
done 
sed -r -i '/^\s*$/d' metrics.csv