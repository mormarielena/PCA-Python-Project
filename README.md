~ World Economy 2021: Multivariate Statistical Analysis
This repository contains a comprehensive data analysis project focused on the 2021 global economic landscape. By utilizing multivariate statistical techniques such as Principal Component Analysis (PCA) and Factor Analysis (FA), the project simplifies complex economic indicators to reveal underlying patterns in national development and sectoral contributions. 

The study analyzes the performance of global economies during 2021, a period characterized by uneven recovery from the COVID-19 pandemic and shifts in global supply chains. The goal is to reduce the dimensionality of economic data and identify the core factors that differentiate the economic profiles of various countries. 


-> Dataset and Indicators
The analysis is based on a dataset of 10 key economic indicators retrieved from Kaggle (global_economy_2021.csv): 

Demographics: Total Population. 

Income: Gross National Income (GNI), GNI Per Capita, and Gross Domestic Product (GDP). 

Sectoral Contributions: Agriculture, Forestry & Fishing; Construction; Transport & Communication; Retail, Trade & Hospitality. 

International Trade: Export and Import values as a percentage of GDP. 

-> Methodologies
1. Data Preprocessing
Missing Value Imputation: Null values were replaced with the mean of their respective columns.

Standardization: Data was scaled using StandardScaler to achieve a mean of 0 and a standard deviation of 1, ensuring equal weight for all variables. 

2. Principal Component Analysis (PCA)
Reduced 10 original variables into significant components. 

The first two components (PC1 and PC2) explain 88.31% of the total variance. 

Selection was guided by the Kaiser Criterion and Cattell’s Scree Plot. 

3. Factor Analysis (FA)

-> Model Validation: 
Bartlett's Test of Sphericity confirmed model validity 

Sampling Adequacy: Kaiser-Meyer-Olkin (KMO) indices were all above 0.7, indicating high suitability for factor analysis. 

Rotation: Applied Varimax rotation to improve factor interpretability. 

-> Key Findings

Economic Scale: The first component/factor effectively captures general economic activity, highly correlated with GDP, GNI, and trade volumes. 

Economic Structure: The second component/factor highlights the distinction between agriculture-dependent economies and those driven by demographics. 

Global Clusters: The analysis identifies distinct country clusters, separating dominant global economies from emerging and agricultural-based nations. 


-> Tech Stack
Language: Python

Data Manipulation: pandas, numpy

Statistical Analysis: scikit-learn (PCA), factor-analyzer (FactorAnalysis, KMO), scipy (Bartlett)

Visualization: matplotlib, seaborn



Developed as part of the "Development and Analysis of Data Systems" (DSAD) curriculum at the Academy of Economic Studies, Bucharest.
