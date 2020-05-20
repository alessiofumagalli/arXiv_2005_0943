# A mathematical model for thermal single-phase flow and reactive transport in fractured porous media

Source code and examples for the paper<br>
"*A mathematical model for thermal single-phase flow and reactive transport in fractured porous media*" by Alessio Fumagalli, Anna Scotti. See [arXiv pre-print](https://arxiv.org/abs/2005.09437).


# Reproduce results from paper
Runscripts for all test cases of the work available [here](./examples).<br>
Note that you may have to revert to an older version of [PorePy](https://github.com/pmgbergen/porepy) to run the examples.

# Abstract
In this paper we present a mathematical model and a numerical workflow for the simula-
tion of a thermal single-phase flow with reactive transport in porous media, in the presence
of fractures. The latter are thin regions which might behave as high or low permeability
channels depending on their physical parameters, and are thus of paramount importance in
underground flow problems. Chemical reactions may alter the local properties of the porous
media as well as the fracture walls, changing the flow path and possibly occluding some
portions of the fractures or zones in the porous media. To solve numerically the coupled
problem we propose a temporal splitting scheme so that the equations describing each phys-
ical process are solved sequentially. Numerical tests shows the accuracy of the proposed
model and the ability to capture complex phenomena, where one or multiple fractures are
present.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv:2005.09437 [math.NA]](https://arxiv.org/abs/2005.09437).

# PorePy version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and revert to commit 254d90e190d20c71b206168c23e7c7ff0ad30735 <br>
Newer versions of PorePy may not be compatible with this repository.

# License
See [license](./LICENSE).
