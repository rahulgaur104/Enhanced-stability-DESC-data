These instructions may not be detailed enough for an unfamiliar reader to reproduce all the results presented in the paper.
If you are having difficulties reproducing these results, please do not hesitate to contact me at rgaur@terpmail.umd.edu or rg6256@princeton.edu

## GS2 analysis of W7-X

To generate Figure 1., we analyzed the attached W7-X equilibrium using the GS2 code. The KBM calculation and ideal ballooning growth rate calculations were performed using a docker image containing GS2 and booz\_xform. The image file can be freely downloaded from [here] (http://hub.docker.com/r/rgaur104/gs2)

Once you have downloaded this image to your cluster/HPC system, run the scripts in the GS2-scripts directory.


## GS2 analysis of the optimized equilibria

Comparison of initial and optimized KBM growth rates is provided in the directory 
GS2\_KBM\_analysis


## ideal-ballooning s-alpha scan of optimized equilibria
can be performed by either creating a python environment with booz\_xform or using the docker image linked above.
The script that perform the s-alpha analysis are provided in the folder s-alpha-3D.

To generate an s-alpha scan of a VMEC file, run

```
python3 s-alpha-ball.py <wout_VMEC_file_name.nc> 0 <some_char>
```

## ideal-ballooning maximum lambda scan
of the initial and optimized equilibria is provided in the directory DESC\_ballooning\_analysis

Also, checkout the DESC tutorial explaining optimization against ideal ballooning instability [here](https://github.com/PlasmaControl/DESC/blob/master/docs/notebooks/tutorials/ideal_ballooning_stability.ipynb)


## Comparison with COBRAVMEC
is provided in the directory COBRAVMEC\_DESC\_comparison

## DESC equilibria and the corresponding scans
are provided in the directory DESC\_BALLOONING\_convergence showing convergece of growth rates with increasing resolution



