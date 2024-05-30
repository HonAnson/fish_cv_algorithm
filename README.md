## Fish Size Estimation Computer Vision Algorithm

### Introduction
Measuring size of object is a challenging task in computer vision. Typical methods include obtaining depth data or using reference object. Specifically, thaere have been machine learning models developed for fish size recgonition based on bio-features of fishes [1].  
In this repository, I showcase an alternative method - stoichastic estimation, for fish size estimation with camera measurement. Which require no depth estimation and reference object, all you need is just a camera.


----

### Quick Start
Creating environment with anaconda (I used python3.10):

```
conda env create -f environment.yml
conda activate fish_cv
python visualization.py
python simulation.py
```
  
If you see something like the following in the projected visualization, its probably due to a fish being too close the camera in camera axis, but . You can rerun `visualization.py` for a better view.

<img src="images/extreme_fish.png" alt="extreme" width="350"/>


---
### Problem Modelling & Assumptions

Imageine you are given a fish recgonition computer vision model, which can effectively identify geomatery of fishes for a given image, as shown in figure below. How could we then estimate size of fishes in a fish farm? 

<img src="demo.gif" alt="demo" width="500"/>



Let's simplify the problem a little bit. We assume the following:
- A 3mx3mx3m fish tank
- 100 fishes in the tank
- camera modelled with projective camera model.
- Near plane and far plane of the camera can be estimated and are constant
- Location of fishes are uniformly distribted in the tank
- Size of fishes follows normal distribution, our goal is to estimate the mean of the distribution
- Fishes rotates randomly

Typically, number of fish and geomatery of fish tank are know in farming conditions. and near and far plane could either be boundaries of the tank or blurring of water.  




We then have the folowing 3d model (the four lines represents fov of the camera):

<img src="images/3d_fish_vid.gif" alt="3d" width="500"/>


The projection onto the camera will be:

<img src="images/projected_fish_vid.gif" alt="3d" width="500"/>




### The algorithm

Now I introduce the algorithm, which is simply the answer to the following question:
```
Can we recover the mean fish size using many measurement of fish size in image?
```

Which I believe is possible with the given assumptions.  

First, consider expected depth in camera frame given the that the fish is seen, in fact, the probability is the volume of the small frustum divided by the bigger frustum. 


<Include drawing of the frustum thing>

What about rotation of the fishes? Let's think deeper about rotation of fishes. If we have a camera viewing a fish as shown in following figure, actually only angle $\theta$ affect the measured size.



Which we can simplify the estimation:



Sadly, have chose to numerically compute the expected size of fish at a depth rotated, as it raises some complicated arithmetics (shown in discussoin).

Which gives the following routine to obtain 



### Discussion

- Occolution
- Consider AI fail rate (angle wise and distance wise)



### Future work












### References 

[1] i-enter coporation, Fish Size Estimation Camera, https://www.i-enter.co.jp/en/marine-tech/fishsize-measurement/