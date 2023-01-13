# Dataset: Movie Lens
# Scripts for training of models: DRR, FairRec and A2Fair
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k --reward-version paper --srm-version paper --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler

# Scripts for ablation study

## A2Fair with Fairec state module
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version adaptative --srm-version paper --srm-size 2 --local-scheduler
