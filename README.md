# HMST

Code for paper **Hyperbolic Multi-semantic Transition for Next POI Recommendation** (WWW 2025 workshop) [(Paper Link)](https://dl.acm.org/doi/abs/10.1145/3701716.3717802)

**Abstract**: The next Point-of-Interest (POI) recommendation has gained significant research interest, focusing on learning users’ mobility patterns from sparse check-in data. Existing POI recommendation models face two main constraints. First, most models are based on Euclidean space and struggle with capturing the inherent hierarchical structures in historical check-ins. Second, various transition semantics in both one-hop and sequential transitions cannot be properly utilized to understand user movement trends. To overcome the above limitations, we introduce rotation operations in hyperbolic space, enabling the joint modeling of hierarchical structures and various transition semantics to effectively capture complex mobility patterns. Specifically, a novel hyperbolic rotation-based recommendation model HMST is developed for the next POI recommendation. To our knowledge, this is the first work to explore the hyperbolic rotations for the next POI recommendation tasks. Extensive experiments on three real-world datasets demonstrate the superiority of our proposed approach over the various state-of-the-art baselines.

**HMST framework**:

<img src="others\HMST_framework.png" alt="image-20250527135413728" style="zoom: 40%;" />

## 1. Environment

Please check `requirements.txt` for the Python packages used in the experiment.

One can create a new Conda environment to run the code as:

```
conda create -n <ENV_NAME> python=3.9 -y
conda activate <ENV_NAME>	
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Please replace `<ENV_NAME>` with any custom environment name.

## 2. Run

1. Please extract `dataset.zip` directly and ensure the `dataset` folder is in the project root directory as:

```
configs/
dataset/
manifolds/
...
```

2. After activating the environment, the experiment code can be run directly using:

```
python main.py --data_name NYC  # run HMST on NYC dataset
python main.py --data_name TKY  # run HMST on TKY dataset
python main.py --data_name CA  # run HMST on CA dataset
```

All configurations for each dataset can be found in `.yaml` files in `configs`.

## 3. Reference

If you find HMST useful in your work, please consider citing our paper:

```
@inproceedings{qiao2025hyperbolic,
  title={Hyperbolic Multi-semantic Transition for Next POI Recommendation},
  author={Qiao, Hongliang and Feng, Shanshan and Zhou, Min and Li, WenTao and Li, Fan},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={1830--1837},
  year={2025}
}
```
