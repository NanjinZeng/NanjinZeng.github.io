---
layout: post
title:  "Two interesting example of instrumental variables"
date:   2019-11-29 20:00:00 +0800
categories: weekly
tag: instrumental variables
---

* content
{:toc}

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
jax: ["input/TeX","output/HTML-CSS"],
displayAlign: "left",
displayIndent: "5em"
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js" type="text/javascript"></script>

Zeng Nanjin WISE IUEC 2016


1.Introduction		{#Introduction}
====================================

Instrumental variable is a common method that we use to find causal relationships, if neither the back-door nor the front-door criterion is satisfied by the variables that we observe. However, it is known to everyone that to find a proper instrumental variable is such a difficult work. After I spent much time in preparing for the application for higher degree, recently I have read two famous paper. Both of them have a ingenious thought using the IV method.


2.examples		{#examples}
====================================

2.1 the mortality rate of European settlers (Acemoglu, Johnson, and Robinson, 2001) {#Acemoglu}
------------------------------------

In AER paper <the Colonial Origins of Comparative Development: An Empirical Investigation> (Acemoglu, Johnson, and Robinson, 2001), they argue that good political institutions is necessary for economy growth in ex-Euro-colonies(for example, Canada versus some Latin-America country). In this topic, a concern used to be that better institutions might be a luxury for those richer countries could enjoy more easily, without using this IV.

>Variable list
>
>IV:  the mortality rate of European settlers
>
>X:  Average Protection against Expropriation Risk (i.e. if the government could expropriate the citizens’ private property easily)
>
>Y: log (GDP per capita)
>
>Z: Latitude, Asia dummy, Africa dummy…

Exogeneity: 
   The high mortal rate which Europeans faces in somewhere is caused by the awful local climate or tropical disease. （e.g. in Latin-America or Africa）This is determined by the natural condition and do not vary by the willing of colonialists. It eases the concern that better institutions might be a luxury for those richer countries could enjoy more easily. It is totally exogenous.

Relevance:
   The high mortality rate would make colonialists hardly settle down. They would build more extractive regimes because they did not care how local people living would be but maximize their profit. (e.g. bad government in some Africa countries). On the contrary, a mild and temperate climate in somewhere like North-America could reduce mortality rate . They could settle down and build some autonomous democratic regimes like Europe, which mind the citizens’ property right.


2.2 countries’ geographic characteristics (Frankel and David, 1999) {#Frankel}
------------------------------------

In AER paper <Does Trade Cause Growth?> (Frankel and David, 1999), they argue that openness and international  trade  could contribute to standards  of living. (i.e. if being open to the world economy will cause a country to be rich.) ). In this topic, a concern used to be that perhaps openness is a luxury that only rich countries can afford, without using this IV.

>Variable list
>
>IV:  countries’ geographic characteristics (i.e. being landlocked/ common border with other countries)
> 
>X:  trade share (i.e. the openness and trade level)
>
>Y: log (GDP per capita)
>
>Z: log(population), log(area)…

Exogeneity: 
   The countries’ geographic characteristics is determined by the natural condition and do not vary by the willing of officials or the wealth of this country. It is totally exogenous.

Relevance:
    The countries’ geographic characteristics would highly affect the countries’ openness level. A landlocked country is difficult to trade with others. 



Reference		{#Reference}
====================================

[1]D Acemoglu, S Johnson, JA Robinson. The colonial origins of comparative development: An empirical investigation. American economic review, 2001


[2]JA FRANKEL, D ROMER. Does Trade Cause Growth?. American Economic Review, 1999 

[3]Jiaming Mao. "Foundations_of_Causal_Inference"[Z].2019-06-09.personal copy
