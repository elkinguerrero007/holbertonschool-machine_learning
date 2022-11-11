# 0x00. Dimensionality Reduction


## **_Resources._** 👌 

 

### **_Read or watch:_**  👈


>> * [Dimensionality Reduction For Dummies — Part 1: Intuition](https://intranet.hbtn.io/rltoken/B9kKbyA71k0cn2CFMTpIfQ)
>> * [Singular Value Decomposition](https://intranet.hbtn.io/rltoken/Qg_s08ni0zOWkqvvRM8ZwQ)
>> * [Understanding SVD (Singular Value Decomposition)](https://intranet.hbtn.io/rltoken/7EODTG3FZ6fQrmZrqkdn5A)
>> * [Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?](https://intranet.hbtn.io/rltoken/WyHO8ZBDqbKmzUoD0Ukf7Q) 
>> * [Dimensionality Reduction: Principal Components Analysis, Part 1](https://intranet.hbtn.io/rltoken/euVIN9M2jJ-PHyOEBnI1lA)
>> * [Dimensionality Reduction: Principal Components Analysis, Part 2](https://intranet.hbtn.io/rltoken/co3YVWGBIdcto2q3HPu51A)
>> * [StatQuest: t-SNE, Clearly Explained](https://intranet.hbtn.io/rltoken/XGKIL0TBES-GY6gO6VoSmg)
>> * [t-SNE tutorial Part1](https://intranet.hbtn.io/rltoken/IaO5r9ba0T_flqHcQv83fA)
>> * [t-SNE tutorial Part2](https://intranet.hbtn.io/rltoken/hariVnyW46RIjyXj6DefGA)
>> * [How to Use t-SNE Effectively](https://intranet.hbtn.io/rltoken/ZGyuMFuDwY6SzE-pM3ZrTw)

# Definitions to skim:
>> * [Dimensionality Reduction](https://intranet.hbtn.io/rltoken/3__-0sq0ymVc6rUhSUF46Q)
>> * [Principal component analysis](https://intranet.hbtn.io/rltoken/-Q1NQBRaQiPLZAlpnXDQoQ)
>> * [Eigendecomposition of a matrix](https://intranet.hbtn.io/rltoken/ZicQZ9TndU2Khb4QLnU9Rg)
>> * [Singular value decomposition](https://intranet.hbtn.io/rltoken/pW3EQwurOaQp4f9SIFXs0w)
>> * [Manifold](https://intranet.hbtn.io/rltoken/W_DWK5vN6rSRqN6jaVe7Ag)
>> * [Kullback–Leibler divergence](https://intranet.hbtn.io/rltoken/EAzyLBFVORoaaWgWc8K9yQ)
>> * [T-distributed stochastic neighbor embedding](https://intranet.hbtn.io/rltoken/EnCpSMJZOJ2E7IMdOof0Jg)

##· **_References_**  👈

>> * [numpy.cumsum](https://intranet.hbtn.io/rltoken/TUz_LerlFe9fPhMuHxJXLg)
>> * [Visualizing Data using t-SNE](https://intranet.hbtn.io/rltoken/2l3jXLWneQVGdNfoXsWMQQ)

>> * [Visualizing Data Using t-SNE](https://intranet.hbtn.io/rltoken/mgNNPvYr_iahfCU8hEZsHQ)

# Advanced:

>> * [Kernel principal component analysis](https://intranet.hbtn.io/rltoken/61bPYClgo7vCg7FHEzSVdQ)
>> * [Nonlinear Dimensionality Reduction: KPCA](https://intranet.hbtn.io/rltoken/34dL3ML5vCExK-iUR9_0Rg)

# Data
>> * [mnist2500_X.txt](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/mnist2500_X.txt)
>> * [mnist2500_labels.txt](https://holbertonintranet.s3.amazonaws.com/uploads/text/2019/10/72a86270e2a1c2cbc14b.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221111T010545Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=f37d1f57fd5f8183fc666af291f2c99287280a14f00a49ee26195454460f65d1)


## **_Built with:_** 🛠️

>> * Ubuntu 20.04 LTS
>> 
>> * Emacs editor && Pycharm
>> 
>> * TensorFlow (version 2.6.0) 
>> 
>> * $ pip install --user tensorflow==2.6
>> 
>> * pycodestyle (version 2.6)
