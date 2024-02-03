WRDIR="question-generation-ptbr"

export WANDB_API_KEY=""
export HF_TOKEN=""
export WANDB_PROJECT="question-generation-ptbr"
cd $WRDIR
pip install --quiet -r requirements.txt
git checkout exps
git pull origin exps

source scripts/run_train.sh experiments/answer-paragraph/t5_base-qg-ap-nopeft.json
source scripts/run_eval.sh