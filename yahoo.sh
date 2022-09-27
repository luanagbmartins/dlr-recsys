rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo/item_groups.pkl
rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo_output_path.json
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo/item_groups.pkl
rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo_output_path.json
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo/item_groups.pkl
rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo_output_path.json
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo/item_groups.pkl
rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo_output_path.json
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler

rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo/item_groups.pkl
rm /home/luana/data/CEIA/Rurax-Moblix/Recommender_system_via_deep_RL/data/yahoo_output_path.json
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --reward-version paper --srm-version paper --srm-size 2 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version adaptative --srm-version adaptative --srm-size 3 --local-scheduler
python -m luigi --module train TrainRS --dataset-version yahoo --train-version yahoo_fair --user-intent-threshold 0.68 --reward-version combining --srm-version combining --srm-size 3 --local-scheduler