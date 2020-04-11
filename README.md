# Stair Alignment Using Deep Learning
#### Published Paper 
To view : [click here](https://www.worldscientific.com/doi/abs/10.1142/S1793351X1940021X)

### Abstract
Mobile robots are widely used in the surveillance industry, for military and industrial applications. In order to carry out surveillance tasks like urban search and rescue operations, the ability to traverse stairs is of immense significance. This paper presents a deep learning based approach for semantic segmentation of stairs, behavioral cloning for star alignment, and a novel mechanical design for an autonomous stair climbing robot. The main objective is to solve the problem of locomotion over staircases with the proposed implementation. Alignment of a robot with stairs in an image is a traditional problem, and the most recent approaches are centred around hand-crafted texture-based Gabor filters and stair detection techniques. However, we could arrive at a more scalable and robust pipeline for alignment schemes. The proposed deep learning technique eliminates the need for manual tuning of parameters of the edge detector, the Hough accumulator, and PID constants. The empirical results and architecture of the stair alignment pipeline are demonstrated in this paper."

### Results
#### Key frames as seen through robot's eyes
![](https://user-images.githubusercontent.com/34411770/74923820-77ca5e00-53f7-11ea-98e8-fc5c1d169025.png)
#### Predicted Actions
![](https://user-images.githubusercontent.com/34411770/74923881-89ac0100-53f7-11ea-9f8b-030d230c99db.png)

####  Raw Result Videos
To view: [click here](https://www.youtube.com/playlist?list=PLflR-cYaxOGFrR4ejMU8Mt5ZF5mPCZNmv)

____

If you end up using this code, or want to cite the above paper, kindly cite as follows. 

```
@article{Panchi2019,
  doi = {10.1142/s1793351x1940021x},
  url = {https://doi.org/10.1142/s1793351x1940021x},
  year = {2019},
  month = dec,
  publisher = {World Scientific Pub Co Pte Lt},
  volume = {13},
  number = {04},
  pages = {497--512},
  author = {Navid Panchi and Khush Agrawal and Unmesh Patil and Aniket Gujarathi and Aman Jain and Harsha Namdeo and Shital S. Chiddarwar},
  title = {Deep Learning-Based Stair Segmentation and Behavioral Cloning for Autonomous Stair Climbing},
  journal = {International Journal of Semantic Computing}
}
```