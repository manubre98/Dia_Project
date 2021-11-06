# Data Intelligence Applications - Online Product Advertising and Pricing with Context Generation

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## Overview

Consider the scenario in which advertisement is used to attract users on an ecommerce website and the users, after the purchase of the first unit of a consumable item, will buy additional units of the same item in future. The goal is to find the best joint bidding and pricing strategy taking into account future purchases.

<p align="center">
    <img src="https://i.imgur.com/4ZO24GA.png" width="600" alt="Scenario"/>
</p>

Imagine a consumable item (for which we have an infinite number of units) and two binary features. Imagine three classes of customers C1, C2, C3, each corresponding to a subspace of the features’ space. Each customers’ class is characterized by:
* a stochastic number of daily clicks of new users (i.e., that have never clicked before these ads) as a function depending on the bid;
* a stochastic cost per click as a function of the bid;
* a conversion rate function providing the probability that a user will buy the item given a price;
* a distribution probability over the number of times the user will come back to the ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).


## Social Influence

* Design of an algorithm maximizing the social influence in every single social network once a budget, for that specific social network, is given. Plot of the approximation error as the parameters of the algorithms vary for every specific network.

<p align="center">
    <img src="https://i.imgur.com/jrBBooS.png" width="1000" alt="vanilla social influence"/>
</p>

* Design of a greedy algorithm such that, given a cumulative budget to perform jointly social influence in the three social networks, finds the best allocation of the budget over the three social networks to maximize the cumulative social influence. Plot of the approximation error as the parameters of the algorithms vary for every specific network.

<p align="center">
    <img src="https://i.imgur.com/F0pqI0q.png" width="300" alt="greedy algorithm"/>
</p>

* Application of a combinatorial bandit algorithm to the situation in which the activation probabilities are not known and we can observe the activation of the edges.  Plot of the cumulative regret as time increases.

<p align="center">
    <img src="https://i.imgur.com/IX2koy7.png" width="700" alt="combinatorial bandit"/>
</p>

## Adding Pricing

* Design of a learning pricing algorithm to maximize the cumulative revenue and application of it, together with the algorithm to make social influence, to the case in which the activation probabilities are known. In doing that, suppose a unique seasonal phase for the whole time horizon. The actual purchase depends on the price charged by the seller and the conversion rate curve. Plot of the cumulative regret.

<p align="center">
    <img src="https://i.imgur.com/vcoBDSC.png" width="850" alt="princing no seasonal"/>
</p>

* Design of a learning pricing algorithm to maximize the cumulative revenue when there are seasonal phases and apply it, together with the algorithm to make social influence, to the case in which the activation probabilities are known. Plot of the cumulative regret.

<p align="center">
    <img src="https://i.imgur.com/qaqfO6f.png" width="850" alt="princing no seasonal"/>
</p>

* Plot of the cumulative regret in the case the seller needs to learn both the activation probabilities and conversion rate curves simultaneously.

<p align="center">
    <img src="https://i.imgur.com/96m8fyl.png" width="400" alt="combinatorial"/>
</p>

## Resources

You can find three Python files and a pdf:
- The pdf file contains the presentation of the project where you can find our final plots and results shown to the professor at the moment of the project submission.
- ```network.py``` contains all the classes that are used to define the graphs used in the experiments.
- ```mab.py``` contains the classes about the learners used for the requests of the project.
- ```main.py``` contains obviously all the code to be run in order to produce the results shown in the presentation; for each point in the requests we run a specific function contained in this file.

Note that for each function called at the end of the file we specify the size and the complexity of the graphs used by the functions. Feel free to change this parameters in order to try different configurations but pay attention to the time to process big and complex networks.


## Team

- Manuel Bressan [[Github](https://github.com/manubre98)] [[Email](mailto:manuel.bressan@mail.polimi.it)]
- Davide Carrara [[Github](https://github.com/davidecarrara98)] [[Email](mailto:davide1.carrara@mail.polimi.it)]
- Filippo Tombari [[Email](mailto:filippo.tombari@mail.polimi.it)]
- Daniela Zanotti [[Github](https://github.com/DanielaZanotti)] [[Email](mailto:daniela1.zanotti@mail.polimi.it)]
