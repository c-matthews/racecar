Algorithms
======================

Different sampling algorithms are available by modifying the ``algo`` argument of the racecar sampler.


Random Walk Metropolis
*****************************

Usable by passing the string ``algo="RWMetropolis"``.

This algorithm does not use any ``params``.


Euler-Maruyama (SGLD)
*****************************

Usable by passing the string ``algo="SGLD"``.

This algorithm does not use any ``params``.

Leimkuhler-Matthews
*****************************

Usable by passing the string ``algo="LM"``.

This algorithm does not use any ``params``.


BAOAB
*****************************

Usable by passing the string ``algo="baoab"``.

- ``g`` (positive float, default 1) :: The friction constant for the Langevin scheme.


Hybrid Monte Carlo (HMC)
*****************************

Usable by passing the string ``algo="hmc"``.

- ``M`` (positive int, default 1) :: Number of steps to take between Metropolization steps.


BADODAB
*****************************

Usable by passing the string ``algo="badodab"``.

- ``g`` (positive float, default 1) :: The friction constant for the Langevin scheme.
- ``mu`` (positive float, default 100) :: The mass for the auxiliary variables.


Racecar
*****************************

Usable by passing the string ``algo="racecar"``.

- ``g`` (positive float, default 1) :: The friction constant for the Langevin scheme.
- ``mu`` (positive float, default 100) :: The mass for the auxiliary variables.
