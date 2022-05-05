# Implemented Short-Range Potentials in DMFF

DMFF already provides the frontends for many frequently-used short-range potentials.
We document them in this page:

## SlaterExForce

Slater-type short-range repulsive force
Ref: jctc 12 3851

Example:
```xml
<SlaterExForce                                                                       
    mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" >
    <Atom type="1" A="1" B="3.977508e+01" />
    <Atom type="2" A="1" B="4.596271e+01" />
    <Atom type="3" A="1" B="4.637414e+01" />
    <Atom type="4" A="1" B="3.831504e+01" />
    <Atom type="5" A="1" B="4.632228e+01" />
</SlaterExForce>                                                                     
```

Formula:

$$ 
\displaylines{
E = \sum_{ij} {A_{ij}P(B_{ij}, r)\exp(-B_{ij}r)} \\
P(B_{ij}, r) = \frac{1}{3}B_{ij}^2 r^2 + B_{ij} r + 1 \\
A_{ij} = A_i A_j \\
B_{ij} = \sqrt{B_i B_j}
}
$$

## QqTtDampingForce

Charge-Charge Tang-Tonnies damping force, used in combination with normal PME

Example:
```xml
<QqTtDampingForce
    mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" > 
    <Atom type="1" B="3.977508e+01" Q="0.14574378"/> 
    <Atom type="2" B="4.596271e+01" Q="0.02178918"/> 
    <Atom type="3" B="4.637414e+01" Q="-0.34690552"/>
    <Atom type="4" B="3.831504e+01" Q="0.00942772"/> 
    <Atom type="5" B="4.632228e+01" Q="0.04964274"/> 
</QqTtDampingForce> 
```

Formula:

$$
\displaylines{
E = \sum_{ij} {- e^{-B_{ij} r} (1 + B_{ij}r) \frac{q_i q_j}{r}} \\
B_{ij} = \sqrt{B_i B_j}
}
$$

## SlaterDampingForce

Slater-type damping function for dispersion. Used in combination with the normal dispersion PME
Ref: jctc 12 3851

Example:
```xml
<SlaterDampingForce
    mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" >
    <Atom type="1" B="3.977508e+01" C6="1.507393e-03" C8="1.241793e-04" C10="4.890285e-06" />
    <Atom type="2" B="4.596271e+01" C6="1.212045e-04" C8="4.577425e-06" C10="8.708729e-08" />
    <Atom type="3" B="4.637414e+01" C6="7.159800e-04" C8="5.846871e-05" C10="2.282115e-06" />
    <Atom type="4" B="3.831504e+01" C6="1.523005e-03" C8="1.127912e-04" C10="4.005600e-06" />
    <Atom type="5" B="4.632228e+01" C6="1.136931e-04" C8="4.123377e-06" C10="7.495037e-08" />
</SlaterDampingForce>
```

Formula:

$$
\displaylines{
E = -\sum_{n=6,8,10} {\sum_{ij} {f_n(x)\frac{\sqrt{C_i^n C_j^n}}{r^n}}} \\
f_n(x) = - e^{-B_{ij} x}\sum_{k=0}^n {\frac{(B_{ij} x)^k}{k!}} \\
x = B_{ij}r - \frac{2(B_{ij}r)^2 + 3B_{ij}r}{(B_{ij}r)^2 + 3B_{ij}r + 3} \\
B_{ij} = \sqrt{B_i B_j}
}
$$
