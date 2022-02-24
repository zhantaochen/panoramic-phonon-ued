## Panoramic mapping of phonon transport from ultrafast electron diffraction and machine learning

This is the code repository for the paper "Panoramic mapping of phonon transport from ultrafast electron diffraction and machine learning" ([https://arxiv.org/pdf/2202.06199.pdf](https://arxiv.org/pdf/2202.06199.pdf)). Please direct any questions about codes to Zhantao (zhantao@mit.edu).

## Key python files
`adjoint_bte/torchdiffeq`
- Modified from `torchdiffeq` of the commit id `4678a03aaeec9ad7daa9888d4e7988d17d011199` [https://github.com/rtqichen/torchdiffeq/tree/4678a03aaeec9ad7daa9888d4e7988d17d011199](https://github.com/rtqichen/torchdiffeq/tree/4678a03aaeec9ad7daa9888d4e7988d17d011199). The major change is to reuse the calculated phonon energy distribution function in the backward pass. This will significantly increase memory usage but provide better numerical stability.

`adjoint_bte/phonon_bte.py`
- defines the forward model of phonon Boltzmann transport equation for a specified layer and material

`adjoint_bte/model_heterostructure.py`
- assemble two layers into a heterostructure with specified boundary and interface conditions

To reproduce results presented in the paper, for synthetic data demonstrations:
- generate synthetic data with the file `Simulation_HeteroStruct.py`, where you can determine several components like materials and interface transmittances (the main target to be learned) and simulate time-dependent atomic mean-squared displacements (MSD) of each layer;
- learn phonon transport properties (i.e., transmittance) with the file `Learn_SyntheticData.py`.

The results corresponds to real UED experiment can be reproduced by directly running `Learn_ExpData.py`.

Please note that we used relatively large batch size (`bs_params`) that utilized roughly 20GB of memories. If you have a GPU with smaller memory size, you may want to reduce the batch size accordingly in order to perform the learning.

## Citing

```
@article{chen2022panoramic,
  title={Panoramic mapping of phonon transport from ultrafast electron diffraction and machine learning},
  author={Chen, Zhantao and Shen, Xiaozhe and Andrejevic, Nina and Liu, Tongtong and Luo, Duan and Nguyen, Thanh and Drucker, Nathan C and Kozina, Michael E and Song, Qichen and Hua, Chengyun and others},
  journal={arXiv preprint arXiv:2202.06199},
  year={2022}
}
```
