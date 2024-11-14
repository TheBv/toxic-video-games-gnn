![GitHub License](https://img.shields.io/github/license/TheBv/toxic-video-games-gnn)
![GitHub release (with filter)](https://img.shields.io/github/v/release/TheBv/toxic-video-games-gnn)
[![Thesis](http://img.shields.io/badge/Thesis-Identifying_toxic_behaviour_in_online_games-fb44ce.svg)](https://publikationen.ub.uni-frankfurt.de/files/81676/Toxic_video_game_classification.pdf)

# Identifying Toxic Video Game Matches with GNN
Repository for the bachelor thesis "Identifying toxic behaviour in online games".
This thesis introduces a way to represent a given video game match as an event graph and using Graph Neural Networks (GNNs) to train a model to detect toxic behaviour in a given match.

More specifically we achieve this by projecting a video game match, which itself can be understood as a temporal network, into an event graph.

This graph we can then enhance using other information such as a graph connecting players that frequently play with eachother.

We can then apply various GNNs on this graph to train a model. More specifically we chose a simple GNN based on [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718).

## Results
Type | Dataset | ROC-AUC
--- | --- | --- 
Multiclass | Detoxify | 0.6134
Multiclass | Annotation | 0.6957
Multiclass | Annotation-Enhanced | 0.7237

**Datasets**


Detoxify: Dataset including 10.000 matches labeling matches as toxic based on the NLP tool [Detoxify](https://github.com/unitaryai/detoxify).

Annotation: Dataset based on roughly 1000 human annotated matches.

Annotation-Enhanced: Dataset based on human annotated matches enhanced with a player graph with weights representing the amount of times they play with eachother.

## Citation
```
@misc{Schrottenbacher2024,
  author    = {Patrick Schrottenbacher},
  title     = {Identifying toxic behaviour in online games},
  institution = {Informatik und Mathematik},
  type      = {bachelorthesis},
  pages     = {35},
  year        = {2024},
  url       = {https://publikationen.ub.uni-frankfurt.de/files/81676/Toxic_video_game_classification.pdf}
  repository = {https://github.com/TheBv/toxic-video-games-gnn}
}
```
