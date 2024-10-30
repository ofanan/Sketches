# Sketches

#### This is an in-work repo, and therefore the documentation is still partial, and the code is not fully stable.
This project studies the usage of sketches and estimators for network telemetry and neural networks.
The project includes efficient Python implementations of the following counters: [Morris](https://www.inf.ed.ac.uk/teaching/courses/exc/reading/morris.pdf), [CEDAR](https://webee.technion.ac.il/~isaac/p/tr11-04_estimators.pdf), [SEAD](https://ieeexplore.ieee.org/document/9537736), [AEE](https://www.researchgate.net/publication/340859493_Faster_and_More_Accurate_Measurement_through_Additive-Error_Counters), [F2P]()
The project provides also Python impelemntation of [Count Min Sketch](https://www.sciencedirect.com/science/article/abs/pii/S0196677403001913), [Space Saving Cache](https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf), and tests that use Resnet and MobileNet modules.
A recommended reading for further details about sketches and counters is:

Li, Shangsen, et al., [A Survey of Sketches in Traffic Measurement:
Design, Optimization, Application and Implementation](https://arxiv.org/pdf/2012.07214.pdf), arXiv preprint arXiv:2012.07214 (2020).

### Overview of main modules
#### F2P, F3P
Floating-Floating counters, described in this [pre-print]().