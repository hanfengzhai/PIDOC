# Simulating dynamics control using neural networks

> **Note:** this is a project developed from the course paper of [MAE 6280](https://classes.cornell.edu/browse/roster/FA20/class/MAE/6280).

## Intro

PIDOC uses [PINNs](https://maziarraissi.github.io/PINNs/) to **simulate** nonlinear dynamics control process, achieved by the encoded control trajectory in the losses. PIDOC has only been used for control the [van der Pol systems](https://www.sciencedirect.com/topics/mathematics/van-der-pols-equation) ([Zhai & Sands, 2022](https://doi.org/10.3390/math10030453)).

## Use

To play with our simple example cases, you need to first download and convert your [tensorflow](https://www.tensorflow.org/) to version [1.x](https://www.tensorflow.org/api_docs/python/tf/compat/v1/); ([Google Colab](https://colab.research.google.com/) is a great platform for it) or directly download it through

~~~
pip install tensorflow==1.15.0
~~~
If you already have ```tf v.2.``` preinstalled, you can considering uninstalling:
~~~
pip uninstall tensorflow==2.7.0
~~~
Then download our repo:
~~~
git clone https://github.com/hanfengzhai/PIDOC.git
~~~

Open the simple case of the van der Pol dynamics:

~~~
cd vanderPol
~~~

Run the basic example case with the benchmark case of 10% added noise:

~~~
python main.py
~~~


If the model start to train, you are ready to go! Try tunning the hyperparameters and change the training data (explore [data](https://github.com/hanfengzhai/PIDOC/tree/main/data)) to play it around! 

* Note that ```main.py``` is only a tutorial template code for playing around, you can also check the [Notebook version](https://github.com/hanfengzhai/PIDOC/blob/main/vanderPol/vanderPol_Fig4_benchmark.ipynb). The uploading of the full version is incomplete and we will finish it soon.

## Constrained optimization

Based on the general framework of PINNs, PIDOC can be used for control all based on the signal-encoded (physics-informed) loss

 *L = MSE<sub>NN</sub> +  MSE<sub>I</sub> + &lambda; MSE<sub>D</sub>*
 
* where *MSE<sub>NN</sub> = MSE(x<sub>prediction</sub>, x<sub>training</sub>)* is the neural network error given the training data.
 
* where *MSE<sub>I</sub> = MSE(x<sub>prediction</sub>(0), x<sub>control</sub>(0))* is the control error given the training data.
 
* where *MSE<sub>D</sub> = MSE(x<sub>prediction</sub>, x<sub>control</sub>)* is the neural network error given the training data.
 
and &lambda; is the Lagrangian multiplier to enforce control (proved in our paper that enlarging which doesn't work for better control).
 
## Limitation

It is clearly stated in the [paper](https://www.mdpi.com/2227-7390/10/3/453): enlarging the systems nonlinearity will cause the reduced control quality. Also, this framework is not for actual experimental control implementation. It is simply a simulation-based idea for using deep learning to model control.

***

### Cite PIDOC

~~~
@Article{math10030453,
AUTHOR = {Zhai, Hanfeng and Sands, Timothy},
TITLE = {Controlling Chaos in Van Der Pol Dynamics Using Signal-Encoded Deep Learning},
JOURNAL = {Mathematics},
VOLUME = {10},
YEAR = {2022},
NUMBER = {3},
ARTICLE-NUMBER = {453},
URL = {https://doi.org/10.3390/math10030453},
ISSN = {2227-7390},
DOI = {10.3390/math10030453}
}
~~~
