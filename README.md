# Hijacking Robot Teams Through Adversarial Communication

This is the code for submitted paper [`Hijacking Robot Teams Through Adversarial Communication`](https://openreview.net/pdf?id=bIvIUNH9VQ) presented as an Oral talk at CoRL 2023.

**Author**: Zixuan Wu, Sean Ye, Byeolyi Han and Matthew Gombolay

The main code to train the adversarial policy is `run_adv_comm_offpolicy` function in `main_heterogeneous.py`. It will:

* Load your pre-trained agent policies in folder `saved_models` .

* Trains surrogate policies to mimic them and adversarial communication policies offline.

* Automatically creates a folder named `models` to save the trained adversarial policies as checkpoints.

The conda environment for running the code can be created using: `conda env create -f environment.yml`. 

This work is still on-going and we will continue refining this repo - the next step includes to train a defender or mainipulate the attacked agents to anywhere we want.

This code is adapted from the `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)` [code](https://github.com/shariqiqbal2810/maddpg-pytorch) with the citation:

```c
@article{lowe2017multi,
  title={Multi-agent actor-critic for mixed cooperative-competitive environments},
  author={Lowe, Ryan and Wu, Yi I and Tamar, Aviv and Harb, Jean and Pieter Abbeel, OpenAI and Mordatch, Igor},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

Please cite our paper with the following format if you find it help:

```c
@inproceedings{wu2023hijacking,
  title={Hijacking Robot Teams Through Adversarial Communication},
  author={Wu, Zixuan and Ye, Sean Charles and Han, Byeolyi and Gombolay, Matthew},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
