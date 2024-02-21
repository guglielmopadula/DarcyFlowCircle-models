# Darcy Flow Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Darcy](https://github.com/guglielmopadula/DarcyFlowCircle) [Flow](https://github.com/guglielmopadula/DarcyFlowCircle) [Dataset](https://github.com/guglielmopadula/DarcyFlowCircle).


The models, with their main characteristics and 
performances, are summed up here.


|Model              |rel u train error|rel u test error| 
|-------------------|-----------------|----------------|
|DeepONet           |0.009            |0.009           |
|DeepONet-reg       |0.005            |0.004           |
|ANO                |0.05             |0.05            |
|VNO                |0.017            |0.018           |
|POD+RBF            |5.4e-07          |0.0004          |
|POD+Tree           |5.4e-07          |0.0004          |