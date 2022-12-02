# SneurIPS

A very simple [snarXiv](http://snarxiv.org/)-like fake ML paper name generator. 

This is basically just a single notebook that trains a GRU based on previous paper titles,
then generates fake ones based on a prompt.

The outputs were originally used for a pub trivia where people try to distinguish real NeurIPS titles from fake ones... they couldn't.

Code samples and the idea for how to do this were shamelessly nabbed from
[this blog post](https://towardsdatascience.com/generating-scientific-papers-titles-using-machine-learning-98c8c9bc637e)
and the Tensorflow tutorial on [text generation with an RNN](https://www.tensorflow.org/text/tutorials/text_generation) so
credit goes to them.


## Installation

Install all requirements in a Python virtual environment using the provided `requirements.txt`.

## Dataset

For the original work, this was trained on the Kaggle dataset [All NeurIPS (NIPS) Papers](https://www.kaggle.com/datasets/rowhitswami/nips-papers-1987-2019-updated?resource=download&select=papers.csv).
Of course you're free to train on anything you like.

