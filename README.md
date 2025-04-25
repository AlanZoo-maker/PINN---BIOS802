# PINN---BIOS802
Modeling Michealis-Menten Enzymes using physics informed neural networks to predict enzyme behavior.

##Software Architecture
This project will use Python in PyCharm Community Edition to model michealis menten enzyme kinectics with machine learning aspects. 

Frameworks: mainly PyTorch will be used, its autograd() function will be employed to quickly calculate gradients.

Link to Benn Moseley's student workshop and initial PINN code implementation for scientific learning:
https://github.com/benmoseley/harmonic-oscillator-pinn-workshop/blob/main/PINN_intro_workshop_student.ipynb


This project mainly uses the same neural network architecture, and the problem setup as physics problem as
defined in Benn Moseley's harmonic oscillator work.

Background:

So how does the code really model a first order chemical rxn; [A] + [B] - k -> [AB]?

Where k is the rate constant, and the concentration of [A] = [B]. We'll also assume that both
reagants are in an aqueous solution where the forward reaction is favored much higher than the reverse reaction;
no equilibrium, completing reactions, or any other fancy reactions are taking place. 

Additionally, we can also assume that [A] and [B] are limited; so both reagants are not undergoing completing
rxns, and are likely binded together through a catalyst/enzyme with the rate constant k. 

In this case a Michealis Menten model of enzyme kinetics can describe this reaction. Here are the assumptions of 
Michealis Menten that are crucial to defining our situation.

1. Enzyme/Catalyst isn't destroyed in the reaction
2. Reverse reaction negligible at early times
3. Steady-State assumption 
4. Free ligand assumption

Without going into the biochemistry of matters too much, all one needes to know for the situation at hands is that 
if there's a certain amount of enzyme, then initially, the rate of reaction is very rapid since most of the enzymes are 
available to process [A] + [B] --> [AB]. However as time increases, the number of enzymes that are available to bind 
decreases, causing the rate of reaction to slow down. 

Here is a mathematical derivation of the chemical equation (trust me):

u(t) = a0(1-1/(1+ k*a0*t))

where u(t) is th concentration of the product, k: rate constant, a0: initial amount of reagant [M],
and t stands for time with any specified unit.

Essentially, as t --> infinity, the amount of product becomes the same as a0, thought the relationship isn't linear.

How do we model this into our PINN to simulate a differential equation?
Essentially, we train a neural network with a added error term that calculates the residual between the exact model 
definition as defined in utils.py, and the PINN model, which will increase in precision over time. 

There is a key point in the way I have defined the problem. During the initial phase of the chemical reaction, the 
volatility is quite high, the first derivative, the rate of reaction, is very high around the ranges of 0 < t < 0.5.

This is predicted by Michealis Menten Kinectics, which essentially tells us that the rate of reaction is very rapid 
initially, then slows down as the reaction progresses. (Steady State Assumption)

Practically speaking, this just means that the loss function of our neural network needs to include an error term for 
both the initial rate of reaction, and the rate of reaction as time progresses. (Line 64 -74.)

As in all neural network models, we are interested in the precision of the model outside of training datasets. 
In our model, we have included a set of generated points a researcher might generate from 0 <t < 0.5. The performance of 
the model outside of these points largely depends on the accuracy of the weights of the PINN, which are influenced by 
real-world phenomena. (In this simple case, this just means that the first two terms of the loss function are penalized 
more heavily than the last two terms, for reasons described above.)

Overall, the neural network learns the model fairly consistently, around 1000 training steps is enough to learn the 
simple reaction.

numpy, matplotlib, pytorch, pandas, and pytest were used in this project.


