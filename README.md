# Hijacking Robot Teams Through Adversarial Communication

This is the code for our submitted paper [`Hijacking Robot Teams Through Adversarial Communication`](https://openreview.net/pdf?id=bIvIUNH9VQ) to CoRL 2023.

**Author**: Zixuan Wu, Sean Ye, Byeolyi Han and Matthew Gombolay

The main code to train the adversarial policy is `run_adv_comm_offpolicy` function in `main_heterogeneous.py`. It will:

* Load the pre-trained policy in folder `saved_models` .

* Train surrogate policies to mimic them and adversarial communication policies offline.

* Automatically create a folder named `models` to save the trained adversarial policies as checkpoints.

This work is still on-going and we will continue refining this repo - the next step includes to train a defender or mainipulate the attacked agents to anywhere we want.

Please cite our paper with the following format if you find it help:

```c
@inproceedings{wu2023hijacking,
  title={Hijacking Robot Teams Through Adversarial Communication},
  author={Wu, Zixuan and Ye, Sean Charles and Han, Byeolyi and Gombolay, Matthew},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
