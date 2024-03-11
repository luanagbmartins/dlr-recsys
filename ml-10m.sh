# Dataset: Movie Lens
# Scripts for training of models: DRR, FairRec and A2Fair

# DRR
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m --reward-version paper --srm-version paper --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m --reward-version paper --srm-version paper --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m --reward-version paper --srm-version paper --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m --reward-version paper --srm-version paper --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m --reward-version paper --srm-version paper --srm-size 3 --local-scheduler

#FairRec
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler

#A2Fair
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

# Scripts for ablation study

## A2Fair with Fairec state module
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version combining --srm-version paper --srm-size 2 --local-scheduler

## A2Fair with Fairec reward
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version combining --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_10m --train-version movie_lens_10m_fair --reward-version paper --srm-version combining --srm-size 3 --local-scheduler

# Generate dataset
# python -m luigi --module src.data.dataset DatasetGeneration --dataset-version movie_lens_10m --local-scheduler
