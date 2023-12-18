# Topology optimization process 

In the current project we want to solve the topology optimisatopn problem for mechanical structures. The reference [paper]([https://github.com/ISosnovik/top](https://www.degruyter.com/document/doi/10.1515/rnam-2019-0018/html)) by Ivan Sosnovik and  Ivan Oseledets. 

Our pipeline for solving the problem: 
-use SIMP method to perform the initial iterations and get the distribution with non-binary densities; 
-use the neural network to perform the segmentation of the obtained image;
-converge the distribution to {0, 1} solution.

We use [TOP dataset](https://github.com/ISosnovik/top) to train the model.
The dataset of topology optimization process. It contains the precise solutions of 10,000 randomly stated problems. Each object is a tensor of shape `(100, 40, 40)`: 100 iterations, `40×40` grid.

In order to optain .h5 file: python prepare_data.py --source TOP4040 --dataset-path ./output_dataset.h5 

To start training the network you need to run: python training_torch.py --dataset-path output_dataset.h5

In our topology optimization process, it is possible to leverage an already trained grid. As shown in methods_results.ipynb, users can utilize a dataset, such as the data stored in output_dataset.h5 from the TOP dataset.To implement this, refer to the methods demonstrated in methods_results.ipynb.

## Some of the key methods identified are:

optimization_step: updating the design variables and computing changes.

optimization: This function executes optimization iterations and saves history.

implement_solution: This method is designed to reapply a solution.

get_history: This function returns the history of previous iterations.

postprocess_history_: This method for post-processing the results of the optimization process, including those derived from a pre-trained grid.

