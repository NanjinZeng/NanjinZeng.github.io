---
layout: post
title:  "Research Proposal(Appendix) Zeng Nanjin"
date:   2021-11-15 23:15:00 +0800
categories: document
tag: economics
---

* content
{:toc}

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

Zeng Nanjin

programme:Master of Economics (quantitative economics)
in National University of Singapore

[to-download-pdf-version-of-these-appendixes](https://github.com/NanjinZeng/Microeconometrics/blob/master/HW/2019-05-05-Piecewise-Linear-Decision-Tree.pdf)

Appendix 1  Literature Review		{#Review}
====================================
A basic assumption is that, the state-owned enterprise is less efficient while the private-owned enterprise is more efficient. Therefore, transition from planned economy to market economy is to move production factors (capital and labor) from state-owned enterprise to private-owned enterprise.
Why did the countries implemented “the shock therapy” have low efficiency in transition and experience recession? From De Melo et al. (1996) and Sachs (1996), scholar began to analyze the efficiency of transition. These two papers found that the liberalization index is positively correlated to economic growth in transition period. The Shock therapy slowed the recession rather than caused the recession. However, according to their later researches, the liberalization index does not have significant correlation with economic growth if they add more control variables (De Melo, 2001).
After that, some scholars tried to analyze this question using empirical approach, including Campos (2000), Campos and Coricelli (2002), Havrylyshyn and Rooden(2003). They gave some political indicators and indexes which are positive correlated to economic growth, but these indicators and indexes are questioned with endogeneity. Popov (2007) found that the distortion of industrial structure before the transition is negative correlated to the economic growth in the transition period. He also finds that China is an outlier, which had high level of distortion of industrial structure and high economic growth rate in the transition period.
On the other hand, some scholars tried to analyze this question using macroeconomic models. For the reason why “The Shock therapy” is not efficient, there are three types of related theories:
Castanheira and Roland (2000) constructed a dynamic general equilibrium model with capital exclusivity assumption. The transition of labor depends on the accumulation of private capital. Thus, there is an optimal speed in transition, based on the speed of saving increase. If the transition is implemented rapidly and private saving is inefficient, the production of economy drops.
The second type of theories tried to give explanation from the prospect of friction in transition. (Atkeson & Kehoe, 1997; Blanchard & Kremer, 1997; Roland & Verdier 1999). The friction in markets determines that the transition should be implemented gradually. For example, Aghion and Blanchard (1994) considered that since the existence of friction in labor market, the state-owned enterprise should not be closed too fast.
Some scholars think that, the transition problem is not from ownership but from distortion of industrial structure. Xu and Lin (2011) constructed a model with partial capital exclusivity assumption. They considered that the government distorted the price of factors (wage rate and interest rate) to subsidy the heavy industry in planned economy. In 1990s, the new government tried to abolish such protective policies in a short time (“The Shock Therapy”). Since the state-owned enterprises (most of them are in heavy industry) were no longer profitable, labor and liquid capital flowed out from this sector. However, non-liquid capital (for example, specialized equipment) was abandoned. Thus, the production of economy drops.
This research is stemmed from these existing researches.



Appendix 2 Some common setting in OLG model		{#OLG}
=====================================
In this economy, there are L identical labors. Each individual survives two period. In the first period, he provides 1 unit of labor inelastically. In the second period, he still needs to consume but does not provide labor.
Each individual maximizes his life time utility,

>$$\text{\ensuremath{\max_{c_{1,t},c_{2,t+1}}U\left(c_{1,t},c_{2,t+1}\right)=\ln\left(c_{1,t}\right)+\frac{1}{1+\rho^{i}}\ln\left(c_{2,t+1}\right)}}$$

ρ is the discount rate. Ci.t is his consumption in period t. The total income of this individual is the sum of his wage in period 1 and interest income in period 2. His budget constraint is 

>$$c_{1,t}+\frac{c_{2,t+1}}{1+r_{t+1}}=w_{t}$$

Each sector in this economy has the same exogenous growing technology. The growth function is Cobb-Douglas form,

>$$Y_{i,t}=A_{t}\left(K_{i,t}\right)^{\alpha}\left(L_{i,t}\right)^{1-\alpha}$$
>$$A_{t+1}=A_{t}\left(1+g_{t}\right)$$
>$$L_{t+1}=L_{t}\left(1+n\right)$$


Appendix 3 Numerical Simulation		{#Simulation}
=====================================
First, I determine the initial value for variables in this OLG model. The initial value will not affect the conclusion of the model, but the final value will have certain numerical difference.

![table1]({{ '/styles/images/table1.jpg' | prepend: site.baseurl  }})

Initial values of macro data are from China Statistical Yearbook 2020. Discount factor and elasticity of capital are referred to reference papers. The wage rate under government control is less than the marginal productivity of labor in 1978 since the government controls the wage rate in a low level to subsidy state-owned enterprise. The wage rate under government control is computed as the average growth rate of wage between 1978 to 2017 from China Labor Statistical Yearbook 2018.



References		{#References}
====================================
Aghion P, Blanchard O J. On the Speed of Transition in Central Europe. NBER Macroeconomics Annual, 1994, 9: 283-320.
  
Atkeson A, Kehoe P. On the Speed of Transition in Neoclassical Benchmark. National Bureau of Economic Research Working Paper Series, 1994, No.6005.
  
Blanchard O, Kremer M. Disorganization[J]. Quarterly Journal of Economics, 1997, 112(4): 1091-1126. 
 
Campos N F. Context Is Everything: Measuring Institutional Change in Transition. Economies. World Bank Working Paper, 2000, No.1809. 
 
Campos N F, Coricelli F. Growth in Transition: What We Know, What We Don’t, and What We Should[J]. Journal of Economic Literature, 2002, 40(3): 793-836. 

Castanheira M, Roland G. The Optimal Speed of Transition: A General Equilibrium Analysis[J]. International Economic Review, 2000, 41(1): 219-239. 

De Melo M. Circumstance and Choice: The Role of Initial Conditions and Policies in Transition Economies. World Bank Working Paper, 2001, No.1866. 

De Melo M, Denizer C, Gelb A. Patterns of Transition from Plan to Market[J]. The World Bank Economic Review, 1996, 10(3): 397-424. 

Havrylyshyn O, Van Rooden R. Institutions Matter in Transition, But So Do Policies[J]. Comparative Economic Studies, 2003, 45:2-24. 

Liao M，Yifu L. Priority development of heavy industry and endogenous formation of planned economic system. Working Paper，2013.

Popov V. 2007. Shock Therapy versus Gradualism Reconsidered: Lessons from Transition Economies after 15 Years of Reforms[J]. Comparative Economic Studies, 2007, 49(1): 1-31.

Roland G, Verdier T. Transition and the Output Fall [J]. Economics of Transition, 1999, 7(1): 1-28. 

Sachs J. The Transition at Mid Decade [J]. American Economic Review, 1996, 86(2): 128-133. 

Xu C, Yifu L. The Development Strategy, the Shock Therapy, and the Economic Transition. Management World, 2011, 1:6-19.