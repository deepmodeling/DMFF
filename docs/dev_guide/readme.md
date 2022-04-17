# About development guide

In this section, you will learn:
    - Architecture of DMFF
    - Code convention
    - Easily expand a new form
    - Write related docs
    - Checklist before PR

DMFF aims to establish a force field calculation and parameter fitting framework supporting automatic differentiation. In order to meet various forms of force field, DMFF calculates energy and force parameters through different force field sub-modules. DMFF ensures sufficient decoupling and modularization at the beginning of the design and can easily add new modules. 

In the *Architecture of DMFF* section, the design of DMFF will be introduced carefully. we will talk about how the parameters are loaded from the XML file and how to organize them for the following calculation and gradient. This work is mainly in charge by a class called `Generator`. Then the calculation code will be explained, which is the bearer of energy and force called `calculation kernel`, a pure function that takes particles' properties and positions and returns the energy. 

Some programming styles and standards should obey in the *Code convention* section. Those conventions will help developers write the code quickly; on the other hand, make the code reviewer and other maintainers easier to modify.

In the *easily expand new form* section, finally, we can write the calculation part of energy and force. However, before you turn your equation to the code to be consistent with other force field sub-modules, this section will introduce the spec that your calculation kernel should follow.

In the *Write related docs* section, we will talk about how to write the manual. Duo to the DMFF is a collection of force field module, each force field has it unique parameters and usage. To make it clear to use and easy to maintain, you should write down the theroy behind the code and the meaning of parameters. 

In the *Checklist before PR* section is what you should do before you publish your work to the Github. In this section, you will know how to write unit test, format checking and proper comment.
