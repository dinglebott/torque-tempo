## About project
**GOAL: Forecast forex price movements with the MOMENT foundation model**\
This is part 4 of the forex prediction project, where I experiment with a pre-trained transformer model by MIT.\
There are 3 model sizes available: `small`, `base`, and `large`. They have parameters of about 40M, 125M, and 385M respectively. For my purposes, I experimented with the 40M model before scaling up, and eventually settled on the ___ model.\
The model is installed from pip, and the output head is fine-tuned on historical forex data.\
*See DOCS.md for detailed results and workflow*\
<br/>
Part 1: [trading-trees](https://github.com/dinglebott/trading-trees), using a tree-based architecture (XGBoost)\
Part 2: [noisy-neurons](https://github.com/dinglebott/noisy-neurons), using neural networks (LSTM + CNN)\
Part 3: [money-meta](https://github.com/dinglebott/money-meta), ensembling models from parts 1 and 2\
<br/>

## Outline of methodology
There were few hyperparameters to be tuned, and most of the experimentation was done by varying the following:
- Feature set
- Learning rate, LR scheduler
- No. of unfrozen transformer blocks
- Class weights
- Batch size, no. of epochs
*See DOCS.md for detailed testing methodology*\
<br/>

## Project structure

<br/>