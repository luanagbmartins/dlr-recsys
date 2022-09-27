rm data/ml-100k/item_groups.pkl
rm data/movie_lens_100k_output_path.json
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm data/ml-100k/item_groups.pkl
rm data/movie_lens_100k_output_path.json
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm data/ml-100k/item_groups.pkl
rm data/movie_lens_100k_output_path.json
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm data/ml-100k/item_groups.pkl
rm data/movie_lens_100k_output_path.json
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm data/ml-100k/item_groups.pkl
rm data/movie_lens_100k_output_path.json
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --user-intent-threshold 0.58 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler