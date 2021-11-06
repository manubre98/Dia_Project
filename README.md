# Data Intelligence Applications - Online Product Advertising and Pricing with Context Generation

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## Overview

Consider the scenario in which advertisement is used to attract users on an ecommerce website and the users, after the purchase of the first unit of a consumable item, will buy additional units of the same item in future. The goal is to find the best joint bidding and pricing strategy taking into account future purchases.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/140611633-49b0e7a4-e293-464a-9d77-420f0947cb33.png" width="400" alt="Scenario"/>
</p>


Imagine a consumable item (for which we have an infinite number of units) and two binary features. Imagine three classes of customers C1, C2, C3, each corresponding to a subspace of the features’ space. Each customers’ class is characterized by:
* a stochastic number of daily clicks of new users (i.e., that have never clicked before these ads) as a function depending on the bid;
* a stochastic cost per click as a function of the bid;
* a conversion rate function providing the probability that a user will buy the item given a price;
* a distribution probability over the number of times the user will come back to the ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).


## General Problem

* Formulate the objective function when assuming that, once a user makes a purchase with a price p, then the ecommerce will propose the same price p to future visits of the same user and this user will surely buy the item. The revenue function must take into account the cost per click, while there is no budget constraint. Provide an algorithm to find the best joint bidding/pricing strategy and describe its complexity in the number of values of the bids and prices available (assume here that the values of the parameters are known). In the following Steps, assume that the number of bid values are 10 as well as the number of price values.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/140612057-5eb1c15b-5ae1-4c27-9de4-1d9f770465fc.jpg"/>
</p>

## Pricing (P3, P4)

* Consider the case in which the bid is fixed and learn in online fashion the best pricing strategy when the algorithm does not discriminate among the customers’ classes (and therefore the algorithm works with aggregate data). Assume that the number of daily clicks and the daily cost per click are known. Adopt both an upper-confidence bound approach and a Thompson-sampling approach and compare their performance.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/140612377-968b435c-b039-429b-9c65-08aabab3495a.jpg" width="850" alt="princing no seasonal"/>
</p>

* Do the same as the step before when instead a context-generation approach is adopted to identify the classes of customers and adopt a potentially different pricing strategy per class. In doing that, evaluate the performance of the pricing strategies in the different classes only at the optimal solution (e.g., if prices that are not optimal for two customers’ classes provide different performance, you do not split the contexts). Let us remark that no discrimination of the customers’ classes is performed at the advertising level.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/140612474-d7c29deb-e6f2-464c-9792-da7db7a803c2.jpg" width="850" alt="princing no seasonal"/>
</p>

## Bidding (P5)

* Consider the case in which the prices are fixed and learn in online fashion the best bidding strategy when the algorithm does not discriminate among the customers’ classes. Assume that the conversion probability is known.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/140612545-98ae55e3-71e2-4ada-949b-7ef2ea599e39.png" width="450" alt="princing no seasonal"/>
</p>

## Pricing & Bidding (P6, P7)

* Consider the general case in which one needs to learn the joint pricing and bidding strategy. Do not discriminate over the customers’ classes both for advertising and pricing.
Then repeat the same when instead discriminating over the customers’ classes for pricing. In doing that, adopt the context structure already discovered.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/140612800-b5b738bf-853c-4986-ba95-b5f7e4850c42.jpg" width="850" alt="princing no seasonal"/>
</p>

## Resources

You can find all the Python files divided for each point and the .pdf of the final report:
- The pdf file contains the presentation of the project where you can find our final plots and all the results obtained.
- ```P3``` and ```P4``` contains all the files related to the Pricing Part.
- ```P5``` contains all the files related to the Bidding Part.
- ```P6``` and ```P7``` contains all the files related to the joint Pricing and Bidding part.

## Team

- Manuel Bressan [[Github](https://github.com/manubre98)] [[Email](mailto:manuel.bressan@mail.polimi.it)]
- Davide Carrara [[Github](https://github.com/davidecarrara98)] [[Email](mailto:davide1.carrara@mail.polimi.it)]
- Filippo Tombari [[Email](mailto:filippo.tombari@mail.polimi.it)]
- Daniela Zanotti [[Github](https://github.com/DanielaZanotti)] [[Email](mailto:daniela1.zanotti@mail.polimi.it)]
