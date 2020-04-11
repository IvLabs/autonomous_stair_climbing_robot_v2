# Training Procedure Codes
Trainer Code for "Deep Learning-Based Stair Segmentation and Behavioral Cloning for Autonomous Stair Climbing" models.

This repository contains the code for the training procedure of the above mentioned paper. The implementation is done in Fast.ai library, since it is very concise and simple to understand. (Earlier code was in Pytorch but then it has been subsequently moved to fastai for simplicity.)

The pipeline in the paper consist of two parts, namely semantic segmentation of stairs and then behavioral cloning for stair alignment. There are separate code files for both the tasks. Script for testing is in the parent directory. Since the paper is not open sourced the authors encourage you to obtain the copy of the paper from this [link](https://www.worldscientific.com/doi/abs/10.1142/S1793351X1940021X).

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
