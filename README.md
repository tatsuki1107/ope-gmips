# Off-Policy Evaluation of Ranking Policies for Large Action Spaces via Embeddings and User Behavior Assumption

# Empirical Evaluation
The experimental script runs in a Docker environment. Please install Docker Desktop according to your opereating system. Then, build the Docker Image.
```
docker compose build
```
## Synthetic Experiments

### How do our estimators perform with varying sample sizes?
```
docker compose run --rm ranking_ope src/synthetic/main_val_size.py eps=0.3 beta=-1.0 variation=val_size
```
### How do our estimators perform with varying the number of unique actions?
```
docker compose run --rm ranking_ope src/synthetic/main_unique_action.py eps=0.3 beta=-1.0 variation=unique_action
```
### How do our estimators perform with varying the length of the ranking?
```
docker compose run --rm ranking_ope src/synthetic/main_len_list.py eps=0.3 beta=-1.0 variation=len_list
```
### How do our estimators perform if Assumption 3.2.(No Direct Effect on Rankings) holds, but Assumption 3.3.(User Behavior Model on Ranking Embedding Spaces) does not hold?
```
docker compose run --rm ranking_ope src/synthetic/main_user_behavior.py eps=0.3 beta=-1.0 variation=user_behavior
```
### How do our estimators perform if even Assumption 3.2.(No Direct Effect on Rankings) does not hold?
```
docker compose run --rm ranking_ope src/synthetic/main_n_unobs_cat_dim.py eps=0.3 beta=-1.0 variation=n_unobs_cat_dim
```

The following is the experimental script in the appendix.
### How do our estimators perform with varying noise levels?
```
docker compose run -rm ranking_ope src/synthetic/main_reward_noise.py eps=0.3 beta=-1.0 variation=reward_noise
```
### How do our estimators perform with varying logging policies?
```
docker compose run --rm ranking_ope src/synthetic/main_beta.py eps=0.3 variation=beta
```
### How do our estimators perform with varying evaluation policies?
```
docker compose run --rm ranking_ope src/synthetic/main_eps.py beta=-1.0 variation=eps
```

In addition to these, you can run the experiments described in the main text for different logging and evaluation policies with `beta=1.0` and `eps=0.8`.
## Real-World Experiment
On the "EUR-Lex4K" dataset
```
docker compose run --rm ranking_ope src/real/main.py dataset_name="EUR-Lex4K" 
```

On the "RCV1-2K" dataset
```
docker compose run --rm ranking_ope src/real/main.py dataset_name="RCV1-2K" 
```


