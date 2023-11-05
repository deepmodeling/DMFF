# 1. Introduction

In the developer's guide, you will learn:
    
+ Architecture of DMFF
+ Code convention
+ Implement a new function form
+ Write docs
+ Checklist before PR

DMFF aims to establish a force field calculation and parameter fitting framework based on the automatic differentiation technique. For different forms of force field, DMFF calculates energy and energy gradients through different sub-modules. DMFF ensures sufficient decoupling and modularization at the beginning of the design so new modules can be added easily. 

In the *Architecture of DMFF* section, the overall design of DMFF will be introduced carefully. We will talk about how the parameters are loaded from the XML file and how they are organized for the following energy and gradient calculations. This work is primarily carried out by the `Generator` class. Then we will explain the heavy-lifting `calculators`, which are pure functions that take atomic positions and force field parameters as inputs and compute the energy.

When developing DMFF, the developers need to obey some programming styles and standards. These are explained in the *Code convention* section. These conventions will help the code reviewers and other developers to understand and maintain the project.

In the *Implementation of New Potentials* section, we will show how to implement a new calculator for a new potential. Before you turn your equations into code, this section will introduce the specs that your code should follow, making it consistent with other force field sub-modules in DMFF.

In the *Document Writing* section, we will talk about how to write docs for your new module. DMFF is a collection of force field calculators, and each force field calculator has its own parameters and may be invoked in different ways. Therefore, to make it easy to use and easy to maintain, the developers are required to document the theroy behind the code and the user interface of the code before PR. 

In the *Checklist before PR* section is what you should do before you submit your PR to Github. In this section, you will learn how to write unit tests, check format, and add proper commentsin your code.

Finally, we will provide a case study, guiding you step-by-step on *how to write a generator*.
+ [An example for developing: how to write a generator?](docs/dev_guide/generator.ipynb)
