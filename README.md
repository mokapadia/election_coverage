# election_coverage
Studying Political Bias and Polarization in Media Coverage of the 2020 U.S. Presidential Election


## Abstract 
In the wake of growing political polarization in the United States and the rapid evolution of technology, particularly in the realms of social media and artificial intelligence, the question of media bias in news coverage of political parties is more pertinent than ever. This report seeks to examine whether American news outlets exhibit bias in their coverage of the 2024 presidential election across political lines. In particular, this report aims to highlight trends concerning the two primary candidates, Donald Trump and Joe Biden. 

Initial exploratory data analysis (EDA) was conducted on a collection of recent news articles, categorized into four groups: Trump, Biden, Election, and Policy, using keywords related to the 2024 U.S. Election. We gathered our article data using an API called NewsAPI and used Mongo as our NoSQL database of choice. The subsequent analysis involved creating visualizations such as pie charts, word clouds, and histograms to depict the distribution of article categories, common vocabulary, source distribution, character counts, and publication dates.
Most news articles were not extremely scored and had a borderline neutral sentiment score.

The core of the analysis focused on the sentiment scores of articles differentiated by the political viewpoint:  left-leaning, right-leaning, and centrist news sources. Centrist new sources act as a baseline of how less politically affiliated new sources should convey their findings. Graphs revealed a very similar negative average sentiment score across both left-leaning sources and right-leaning counterparts. Centrist scores were scored around the same magnitude but in a positive direction. Often these averages were skewed by more extremely scored sources within these classifications. Detailed sentiment comparisons between specific news outlets on specific categories failed to depict strong trends from expected differing viewpoints on categories like President Biden or the election in general. The specific findings between sources indicate nuanced differences in how news outlets cover political figures and election-related topics. ANOVA analysis depicted that there were no significant differences among the average scores of sources within each political viewpoint. 

We concluded that these sentiment score differences amongst politically-affiliated sources were not significant, and therefore, media bias on election was not detected on partisan lines. Some limitations of our analysis include limited content to score these articles and sentiment score’s inability to capture subjective biases. 

To check out the full [report](report and presentation/Report.pdf)


