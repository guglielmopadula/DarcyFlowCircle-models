# Darcy Flow Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Darcy](https://github.com/guglielmopadula/DarcyFlowCircle) [Flow](https://github.com/guglielmopadula/DarcyFlowCircle) [Dataset](https://github.com/guglielmopadula/DarcyFlowCircle).


The models, with their main characteristics and 
performances, are summed up here.


|Model              |rel u train error|rel u test error| 
|-------------------|-----------------|----------------|
|DeepONet           |4.9e-01          |4.8e-01         |
|DeepONet-reg       |1.6e-01          |1.4e-01         |
|ANO                |2.0e-00          |1.8e-00         |
|VNO                |2.2e-01          |2.1e-01         |
|EIMPoints+RBF      |3.2e-01          |3.7e-01         |
|EIMPoints+TreeReg  |3.2e-01          |4.0e-01         |
|POD+RBF            |3.2e-01          |3.7e-01         |