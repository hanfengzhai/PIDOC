# [Physics-Informed Deep Operator Control](https://arxiv.org/abs/2112.14707)
Physics-Informed Deep Operator Control (PIDOC), a deep learning method for controlling nonlinear chaos.

![schematic view of Physics-Informed Deep Operator Control](/doc/PINC_schematic.jpg)

* **What is PIDOC?**

PIDOC is based on [PINNs](https://maziarraissi.github.io/PINNs/), is a general framework can be used for control nonlinear dynamics, achieved by the encoded control trajectory in the losses. Right now, PIDOC has only been used for control the [van der Pol systems](https://www.sciencedirect.com/topics/mathematics/van-der-pols-equation). However, if carefully tuned, it can be applied to more complex systems.

* **How to use PIDOC?**

***

If you use our package please considering cite:
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
~~~
