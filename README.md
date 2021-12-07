# DRL Recsys

## Estrutura

- data
- model
- notebook
- src
    - data
    - environment
    - model
    - train_model
    - recsys_fair_metrics

## Experimentos

### DRR
- Alterar os parâmetros desejados em `model/movie_lens_100k.yaml`
- Rodar o treinamento:
    `python -m luigi --module train DRLTrain --dataset-version movie_lens_100k --train-version movie_lens_100k --use-wandb --local-scheduler`

### FairRec
- Alterar os parâmetros desejados em `model/movie_lens_100k_fair.yaml`
- Rodar o treinamento:
    `python -m luigi --module train DRLTrain --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --use-wandb  --local-scheduler`

### Avaliação
- Para realizar a avaliação no ambiente simulado:
    `notebooks/evaluation_online.ipynb`

# Resultados

# Referências

Singh, A., & Joachims, T. (2018). **Fairness of Exposure in Rankings.** Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

Patil, V., Ghalme, G., Nair, V.J., & Narahari, Y. (2020). **Achieving Fairness in the Stochastic Multi-armed Bandit Problem.** ArXiv, abs/1907.10516.

Liu, F., Tang, R., Li, X., Ye, Y., Chen, H., Guo, H., & Zhang, Y. (2018). **Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling.** ArXiv, abs/1810.12027.

Liu, W., Liu, F., Tang, R., Liao, B., Chen, G., & Heng, P. (2020). **Balancing Between Accuracy and Fairness for Interactive Recommendation with Reinforcement Learning.** Advances in Knowledge Discovery and Data Mining, 12084, 155 - 167.

https://coggle.it/diagram/YVRpbgDOVWelsxPQ/t/fairness
