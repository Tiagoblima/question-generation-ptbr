BASE_DIR="./data/du-squadv1/"
REF_FILE="$BASE_DIR/tgt-test.txt"
SRC_FILE="$BASE_DIR/src-test.txt"
rm -rf metrics.csv
echo "Bleu_1,Bleu_2,Bleu_3,Bleu_4,ROUGE_L,model" > metrics.csv
for model_pred in $(ls "$BASE_DIR/preds" ); do
    echo "Evaluating $model_pred"
    python src/qgevalcap/eval.py --out_file "$BASE_DIR/preds/$model_pred"  \
                    --tgt_file $REF_FILE \
                    --src_file $SRC_FILE | tail -n 2 >> metrics.csv
done 
sed -r -i '/^\s*$/d' metrics.csv