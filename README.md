# Weighted Low-Rank Approximation for Transformer Compression.
**Final project for Skoltech NLA-2022 course.**

**Team members:** Seleznyov Mikhail, Galichin Andrey, Kovaleva Maria.

There are various ways to compress large language models. 
One of them is good old SVD, applied to weight matrices in linear layers. 
However, it treats all parameters equally.

Authors of paper [Large model compression with weighted low-rank factorization](https://arxiv.org/pdf/2207.00112.pdf)
hypothesized, that some parameters are more important then others, and should be reconstructed more accurately. 
To address that, one could somehow compute the importance of each parameter and then use weighted low-rank approximation.
A sane approach for parameter importance estimation is to use Fisher information matrix.

Unlike usual low-rank approximation, weighted low-rank approximation in general case does not have a closed-form solution.
In the paper authors made a simplifying assumption, that all parameters in one row of each weight matrix have the same importance, 
computed as a mean of corresponding row in Fisher information matrix. With that the task again reduces to applying SVD.

However, there are some iterative algorithms for weighted low-rank approximation, 
for example, described in papers [Weighted Low-Rank Approximations](https://www.aaai.org/Papers/ICML/2003/ICML03-094.pdf) 
and [Weighted Low-Rank Approximation and Acceleration](https://arxiv.org/pdf/2109.11057.pdf).
So we decided to check, if more accurate solution of weighted low-rank approximation helps to compress language models more efficiently.
