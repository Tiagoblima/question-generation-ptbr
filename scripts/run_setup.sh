WRDIR="question-generation-ptbr"
WANDB_API_KEY=""
HF_TOKEN=""
cd $WRDIR
pip install --quiet -r requirements.txt
git checkout exps
git pull origin exps

echo   "export WANDB_API_KEY=$WANDB_API_KEY"  > run_setup.sh
echo   "export HF_TOKEN=$HF_TOKEN" >> run_setup.sh
echo   "export WANDB_PROJECT=question-generation-ptbr \n" >> run_setup.sh
echo   "source scripts/run_train.sh experiments/answer-paragraph/t5_base-qg-ap-nopeft.json" >> run_setup.sh
echo   "source scripts/run_eval.sh tiagoblima/t5_base-qg-ap-nopeft paragraph,answer" >> run_setup.sh