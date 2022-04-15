# About development guide

In this section you will learn:
    - Architecture of DMFF
    - Code convention
    - How frontend works
    - How backend works
    - Easily expand new form
    - Write related docs
    - Checklist before PR

DMFF aims to establish a force field calculation and parameter fitting framework supporting automatic differentiation. In order to meet various forms of force field, DMFF calculates energy and force parameters through different force field sub-modules. DMFF ensures sufficient decoupling and modularization at the beginning of the design and can easily add new modules. 

In the *Architecture of DMFF* section, the design of DMFF will be introduced carefully. 

In the *Code convention* section, some programming styles and standards should obey. Those conventions on the one hand will help developers to write the code quickly; on the other hand make the code reviewer and other maintainers easier to modify.

In the *How frontend works* section, we will talk about how the parameters loaded from XML file and how to origanized them for the next calculation and gradient. This work are mainly in charged by a class called `Generator`.

In the *How backend works* section, we will explain principle of the calculation module. The bearer of energy and force is called `calculation kernel`, which is a pure function that takes particles properties and positions and return the energy. 

In the *easily expand new form* section, finally we can write the calculation part of energy and force. However, before you turn your equation to the code, in order to consistent with other force field sub-modules, this section will introduce the spec that your calculation kernel should follow.


