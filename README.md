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
In order to optain .h5 file: python prepare_data.py --source TOP4040 --dataset-path ./output_dataset.h5 

To start training the network you need to run: python training_torch.py --dataset-path output_dataset.h5

In our topology optimization process, it is possible to leverage an already trained grid. As shown in methods_results.ipynb, users can utilize a pre-trained grid, such as the data stored in output_dataset.h5 from the TOP dataset.To implement this, refer to the methods demonstrated in methods_results.ipynb.

Some of the key methods identified are:
optimization_step: updating the design variables and computing changes.
optimization: This function executes optimization iterations and saves history.
implement_solution: This method is designed to reapply a solution.
get_history: This function returns the history of previous iterations.
postprocess_history_: This method for post-processing the results of the optimization process, including those derived from a pre-trained grid.
