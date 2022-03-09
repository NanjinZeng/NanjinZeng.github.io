---
layout: post
title:  "Automatic Accountant Assistant User Manual"
date:   2022-03-09 14:15:00 +0800
categories: document
tag: pyspark
---

* content
{:toc}

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

Zeng, Nanjin  
master student of 20/21 | Quantitative Economics | National University of Singapore

[to-download-original-code-of-this-app(.ipynb)](https://github.com/NanjinZeng/Pyspark_test/blob/main/AAA%20version%201.0%20.ipynb)

1.Introduction		{#Introduction}
====================================
Automatic accountant assistant is an accounting entries collection and analysis system for commercial companies (powered by pyspark). Operator can use it to record accounting entries, to generate income statements, to perfrom revenue/expense analysis and to interact with user-defined budget.



2.Funtions		{#Funtions}
=====================================
Logged in with different indentity, operator can use indentity-related functions of this system.

2.1 Administrator (mode 1)

Operator can enjoy full features of this system.

1. regist or remove multiple new user accounts
2. log in new entries to General Ledger/delete certain entries from General Ledger
3. generate the income statement/compare the financial status for specified months and receive coresponding advice
4. set or reset budget for specified month
5. compare the actual revenue/expense to budget for specified month and receive coresponding advice

2.2 User (mode 2)

Operator can enjoy limited features of this system.

1. log in new entries to General Ledger
2. generate the income statement
3. compare the actual revenue/expense to budget for specified month and receive coresponding advice

P.S.

+ User can not access to user account management module. (system security)
+ User can not delete certain entries from General Ledger. (internal control requirement)
+ User can not modify budget. (internal control requirement)
+ User can not compare the financial status for specified months. (limit information access)
+ Administrator can also log in as user 0. (In this case, features will be limited.)




3.User Information Initialization (mode 0)		{#Initialization}
=================================================
After being verified with producer-pre-defined master key. Operator can reset the administrator password and delete all users' account.

+ General Ledger and Budget information will not be affected.