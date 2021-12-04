# dlr-recsys


## DRR
- Alterar os parâmetros desejados em `model/movie_lens_100k.yaml`
- Rodar o treinamento:
    `python -m luigi --module train DRLTrain --dataset-version movie_lens_100k --train-version movie_lens_100k --local-scheduler`

## FairRec
- Alterar os parâmetros desejados em `model/movie_lens_100k_fair.yaml`
- Rodar o treinamento:
    `python -m luigi --module train DRLTrain --dataset-version movie_lens_100k --train-version movie_lens_100k_fair --local-scheduler`

## Avaliação
- Para realizar a avaliação no ambiente simulado:
    notebooks/evaluation_online.ipynb
