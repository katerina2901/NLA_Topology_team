# Topology optimization process 
We use [TOP dataset](https://github.com/ISosnovik/top) to train the model.
The dataset of topology optimization process. It contains the precise solutions of 10,000 randomly stated problems. Each object is a tensor of shape `(100, 40, 40)`: 100 iterations, `40×40` grid.

```latex
@article{sosnovik2017neural,
  title={Neural networks for topology optimization},
  author={Sosnovik, Ivan and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1709.09578},
  year={2017}
}
```