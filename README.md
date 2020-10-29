# Auto_DIff
This is the repo for SI211 Numeral Analysis in ShanghaiTech University mainly implementing automatic differentiation

## description
This project contains four main modules. 
* The **ForwardDiff** implements the forward mode of AD.
* The **BackwardDiff** implements the backward mode of AD. 
* The **ADDiff** combines the forward mode and backward
mode, and therefore, there are many unnecessary operations. Please use the single module if you do not need to obtain the results of
the two methods at the same time.
* The **Numerical_diff** implements numerical difference with five points formula. 

**So far, only the combination of the atom operations +,−,∗,/,sin,cos are supported.**

For more details, please refer to [report.pdf](report.pdf).

## performance
the time-consuming comparisons are:

|method |time(s)|
|:---:|:---:|
|df| **0.067**|
|ADforward| 63.538 |
|ADbackward|0.286|
|Numerical diff|64.777 |

the accuracy comparisons are:

$$ acc_i =  ||res_i − res_{df}||_2^2 $$ 

| method |  acc|
|:---:|:---:|
|ADforward| 1.561 × 10−17|
|ADbackward| **1.557 × 10−17**|
|Numerical diff|1.775 × 10−11|

the results can be reproduced by the scripts in folder [script](script).
