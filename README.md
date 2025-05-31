# Off-Policy Evaluation of Ranking Policies via Embedding-Space User Behavior Modeling
This repository provides the experimental code for the paper titled "Off-Policy Evaluation of Ranking Policies via Embedding-Space User Behavior Modeling".

## Package Version
The versions of Python and its associated libraries are as follows:
```
[tool.poetry.dependencies]
python = ">=3.9,<3.10"
obp = "^0.5.5"
numpy = "^1.22.5"
scikit-learn = "1.1.3"
scipy = "1.9.3"
pandas = "^1.3.2"
seaborn = "^0.11.2"
matplotlib = "3.7.3"
hydra-core = "^1.3.2"
tqdm = "^4.66.2"
```


## Run Experiments
The experimental script runs in a Docker environment. Please install [Docker Desktop](https://docs.docker.com/desktop/) according to your opereating system. Then, build the Docker Image.
```
docker compose build
```
### 4. Synthetic Experiments
The CSV and plot image files of the experimental results are outputted to the `./src/synthetic/logs` directory.

**Performance of our proposed estimators with varying sample sizes**
```
docker compose run --rm ranking_ope src/synthetic/main_val_size.py variation=val_size
```

**Performance of our proposed estimators with varying number of unique actions**
```
docker compose run --rm ranking_ope src/synthetic/main_unique_action.py variation=unique_action
```

**Performance of our proposed estimators with varying length of the ranking**
```
docker compose run --rm ranking_ope src/synthetic/main_len_list.py variation=len_list
```

**Performance of our proposed estimators if Assumption 3.2. does not hold**
```
docker compose run --rm ranking_ope src/synthetic/main_n_unobs_cat_dim.py variation=n_unobs_cat_dim
```

**(Appendix F)**

**Performance of our proposed estimators with varying noise levels**
```
docker compose run -rm ranking_ope src/synthetic/main_reward_noise.py variation=reward_noise
```

**Performance of our proposed estimators with varying logging policies**
```
docker compose run --rm ranking_ope src/synthetic/main_beta.py variation=beta
```

**Performance of our proposed estimators with varying target policies**
```
docker compose run --rm ranking_ope src/synthetic/main_eps.py variation=eps
```

**Performance of our proposed estimators with varying number of deficient unique actions**
```
docker compose run --rm ranking_ope src/synthetic/main_n_deficient_actions.py variation=n_deficient_actions
```

**Performance of our proposed estimators if Assumption 3.3 does not hold while Assumption does hold**
```
docker compose run --rm ranking_ope src/synthetic/main_user_behavior.py variation=user_behavior
```

**Performance of our proposed estimators using SLOPE, which selects the optimal embedding dimension**
```
docker compose run --rm ranking_ope src/synthetic/main_embed_selection.py variation=embed_selection
```

### 5. Real-World Data Experiments
At first, please download the EUR-Lex4K and RCV1-2K datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). Then, rename the train and test txt files to “train.txt” and “text.txt”, respectively, and store them in `./rawdata/EUR-Lex4K/` and `./rawdata/RCV1-2K/`.  

The CSV and plot image files of the experimental results are outputted to the `./src/real/logs` directory.  

**On the "EUR-Lex4K" dataset**
```
docker compose run --rm ranking_ope src/real/main.py dataset_name="EUR-Lex4K" 
```

**On the "RCV1-2K" dataset**
```
docker compose run --rm ranking_ope src/real/main.py dataset_name="RCV1-2K" 
```


