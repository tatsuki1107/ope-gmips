# Off-Policy Evaluation of Ranking Policies for Large Action Spaces via Embeddings and User Behavior Assumption

# Empirical Evaluation
The experimental script runs in a Docker environment. Please install Docker Desktop according to your opereating system. Then, build the Docker Image.
```
docker compose build
```
## Synthetic Experiments

### How do our estimators perform with varying sample sizes?
```
docker compose run ranking_ope poetry run python src/synthetic/main_val_size.py eps=0.3 beta=-1.0 variation=val_size
```
### How do our estimators perform with varying the number of unique actions?
```
docker compose run ranking_ope poetry run python src/synthetic/main_unique_action.py eps=0.3 beta=-1.0 variation=unique_action
```
### How do our estimators perform with varying the length of the ranking?
```
docker compose run ranking_ope poetry run python src/synthetic/main_len_list.py eps=0.3 beta=-1.0 variation=len_list
```
### How do our estimators perform if the assumption of No Direct Effect on Ranking holds, but the assumption of User Behavior on Ranking Embedding Spaces does not hold?
```
docker compose run ranking_ope poetry run python src/synthetic/main_user_behavior.py eps=0.3 beta=-1.0 variation=user_behavior
```
### How do our estimators perform if even the assumption of No Direct Effect on Ranking does not hold?
```
docker compose run ranking_ope poetry run python src/synthetic/main_n_unobs_cat_dim.py eps=0.3 beta=-1.0 variation=n_unobs_cat_dim
```

The following is the experimental script in the appendix.
### How do our estimators perform with varying noise levels?
```
docker compose run ranking_ope poetry run python src/synthetic/main_reward_noise.py eps=0.3 beta=-1.0 variation=reward_noise
```
### How do our estimators perform with varying logging policies?
```
docker compose run ranking_ope poetry run python src/synthetic/main_beta.py eps=0.3 variation=beta
```
### How do our estimators perform with varying evaluation policies?
```
docker compose run ranking_ope poetry run python src/synthetic/main_eps.py beta=-1.0 variation=eps
```
### How data-driven embedding selection affects the performance of our estimators?
```
docker compose run ranking_ope poetry run python src/synthetic/main_embedding_selection.py eps=0.3 beta=-1.0 variation=embedding_selection
```
### How do our estimators using marginalized importance weight estimation parform with varying sample sizes?
```
docker compose run ranking_ope poetry run python src/synthetic/main_weight_estimation.py eps=0.3 beta=-1.0 variation=weight_estimation
```
In addition to these, you can run the experiments described in the main text for different logging and evaluation policies with `beta=1.0` and `eps=0.8`.
## Real-World Experiment
```
docker compose run ranking_ope poetry run  python src/real/main.py 
```


