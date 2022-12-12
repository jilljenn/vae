# Variational Factorization Machines

Presented at IEEE BigData 2022 in December 2022.

- [Read the article](http://jiji.cat/bigdata/vie2022vfm.pdf)
- See [slides](https://jjv.ie/slides/vfm.pdf) presented at IEEE BigData 2022
- See [introductory slides](https://jjv.ie/slides/vfm-kyodai.pdf) for a presentation given at Kyoto University

You can cite the article as:

> Vie, J. J., Rigaux, T., Kashima, H. (2022, December). **Variational Factorization Machines for Preference Elicitation in Large-Scale Recommender Systems.** In *Proceedings of IEEE BigData 2022*, in press.

    @inproceedings{Vie2022VFM,
        Author = {{Vie}, Jill-J{\^e}nn and Rigaux, Tomas and {Kashima}, Hisashi},
        Langid = {english},
        Title = {Variational Factorization Machines for Preference Elicitation in Large-Scale Recommender Systems},
        Booktitle = {Proceedings of IEEE BigData 2022, in {press}},
        Url = {https://jiji.cat/bigdata/vie2022vfm.pdf},
        Year = 2022}

## Implementations

- `vfm.py` needs TensorFlow <2.0, for example 1.15. See `requirements-old.txt`
- `vfm-torch.py` is a reimplementation in PyTorch. See `requirements-torch.txt`
- `vfm-tomasrch.py` is another implementation in PyTorch with closed form for the regression case. No sampling is made.

## Notice

This repository was previously collecting implementations of VAE. Notably we can see that since TF 2.0, TF really looks like Keras so it becomes cumbersome to define VAEs (needs TensorFlow Probability layers to get a short boilerplate code). Also surprisingly, TF 1.15 distributions are faster than PyTorch distributions.
