# Dataset: Yahoo
# Scripts for training of models: DRR, FairRec and A2Fair
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo --reward-version paper --srm-version paper --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler

# Scripts for ablation study

## A2Fair with Fairec state module
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version adaptative --srm-version paper --srm-size 2 --local-scheduler
