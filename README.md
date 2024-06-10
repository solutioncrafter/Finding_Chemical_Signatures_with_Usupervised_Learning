
# Finding Chemical Signatures with Unsupervised Learning

For the complete analysis open the following [notebook](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/blob/master/notebooks/Finding_Chemical_Signatures_with_Unsupervised_Learning.ipynb)


-------

# 1. Introduction


In materials sciences, the chemical structure of a material can be determined by measuring its interaction with light. Different frequencies of light (colors) will interact differently with every material, becoming features of its chemical structure. A single observation will be then an n-dimensional array, where each array element is the measured intensity of a frequency. In the domain of materials science, every observation or data point, is a spectrum. 

This notebook shows the use of unsupervised learning to extract latent (hidden) chemical signatures from a dataset describing a chemical process. The dataset is a sequence of Raman spectra measured during the formation of a single crystal of glycine from a solution. In simple words, the birth of a crystal. 

This notebook is inspired in the following published paper: https://doi.org/10.1073/pnas.2122990119


# 2. Data Description


## Dataset Overview

- The dataset is a sequence of Raman spectra from a glycine crystallization process. 
- It is presented as numerical tabular data, with 990 features and 73 instances. Each data point, or spectra, is an array with 990-dimensions.
 
- Source of dataset: DOI: **https://doi.org/10/gp9f7w**

## Data Dictionary

- Each feature is a Wavenumber (frequency) in the Raman spectrum. 

- The first column is the time tag at which the data point was recorded.

## Reformatting the Dataset

- The data was reformatted from its original structure to make it easier to import as a dataframe, with this [Python script.](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/blob/master/src/transform_data.py)
- A new column with timestamps was added according to the measurement frequency provided by the paper.



# 3. Problem Statement and Objectives


## Problem Statement

In materials science, understanding the chemical structure of a material involves analyzing its interaction with light, which generates spectral data. We need to uncover hidden chemical signatures from Raman spectra data collected during the crystallization of glycine from a solution.

## Objectives 

1) **Determine the Optimum Number of Chemical Signatures:** Identify the optimal number of distinct spectra that best describe the chemical process during crystallization.
2) **Extract Main Chemical Signatures:** Isolate the primary chemical signatures representing different chemical species involved in the crystallization process, with a focus on the nucleation phase.
3) **Describe the Behavior of Each Component:** Analyze and describe how each identified component behaves throughout the crystallization process.

4) **Ensure Interpretability:** Ensure that the latent chemical signatures extracted from the data are interpretable and provide clear insights into the chemical process.


# 4. Data Cleaning and Pre-Processing

## Filter Random Noise and Low Variance Components 

- First we use Singular Value Decomposition (SVD) to understand how many components carry most of the variance.

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/156a7eca-1be2-46c0-a3fd-ab992b47b1f3)


**Insight:**  80% of the variance is in just 10 components. 

- Setting a 80% threshold, filtering is performed by truncated SVD: Truncating at 10 components and performing a reconstruction. 

# 5. Exploratory Data Analysis (EDA)


## Visualize the Normalized Dataset 

- A heatmap of the dataset helps visualize how different parts of the spectrum show different intensity features during different states:
    1)  The features are broad while the sample is in a liquid state (solution).
    2)  There is a distinct and short transition region which matches the nucleation or birth of a crystal.
    3)  The final state is characterized by sharp intensity features when the sample reaches the solid state (crystal).

- Red lines are drawn at meaningful times, from where 2D plots of the spectra are later extracted.

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/2c620a33-bfb9-48a4-a861-9ac31a486442)



## Visualize Key Spectra
- Data points (spectra) at the marked red lines are extracted for visualization.

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/0fbc3068-f17c-49f3-9070-a5ccf83c04a4)


- By inspecting the data, there seems to be **at least 2** evident chemical signatures from the initial and final state. 

# 6. Clustering

- A Hierarchical clustering algorithm is used to understand the different classes of chemical signatures present in the dataset.

- Due to the rapid nature of the crystallization process, it may result in a strong class imbalance. Hierarchical clustering was selected for this reason. 


## Finding The Optimum Number of Clusters 
- A set of evaluation scores are computed and plotted for various cluster numbers to discover the optimum value:

    - Within-cluster sum of squares (WCSS) or "Elbow Curve": Where the optimum number of clusters is at the inflection point
    - Silhouette Score: We look for the maximum value
    - Calinski Harabasz Score: We look for the maximum value
    - Davies Bouldin Score: We look for the minimum value

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/8958f564-ced3-4c6b-9994-a8d5dd9998f6)


**Insight:**  All curves, except the Calinski Harabasz, show that 3 clusters is the best choice.


## Visualization of Hierarchical Clustering

- Hierarchical clustering is used to cluster the data in 3 clusters.
- The original dataset is reduced to a 2-dimensional space by Principal Component Analysis (PCA).
- Clusters are visualized on the 2-dimensional representation by different colors. 

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/870565c2-f182-45cd-b321-5d6b631bf685)


**Insights:**

- The visualization helps confirm the existence of 3 cluster classes.

- Few points belong to class 3 which belong too the period of nucleation ("crystal birth")

 
# 8. Dimensionality Reduction

- 3 cluster classes of distinct chemical signatures were already found. A complementary method is needed to extract the chemical signatures, that will be later used to understand the chemical structures.

- Along with reducing the number of dimensions or features of a dataset,  dimensionality reduction methods can be used to find the principal components or basis vectors describing its main variability. These methods can be used to separate the chemical signatures that describe the chemical process represented by the dataset. 


## Non-Negative Matrix Factorization

- NMF was chosen to maintain the interpretability: Raman spectra can only take non-negative values. 

- The number of components was set to 3 from the previous clustering results. 


## Visualize NMF Reduced Representation (Weights)


- Visualize normalized 3-dimensional reduced representation: Component weights extracted from NMF as they evolve through time.
- Each 3-dimensional data point (set of weights) is normalized so it adds up to 1 and plotted for better interpretability.

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/b24defa9-82f7-4d1c-8869-5bb87c389da5)



**Insight:** Plotting the reduced representation through time help us understand how the chemical signatures evolve over time and gives valuable information to the researchers about the nature of the process. 


## Visualize NMF Components (Raman Chemical Signatures)
- The 3 components extracted by NMF are plotted.

- The components, or basis vectors, are the principal chemical signatures describing the process (Raman spectra).

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/8dab6ab9-d037-4d24-9912-5067cef58704)



**Insights:** 

- The extracted components, when plotted against the wavenumbers, can be interpreted as Raman spectra. 
- Each of these components are signatures of a chemical species with certain structure. 

- These extracted Raman spectra will be used by materials specialists to understand the chemical changes during the formation of a crystal.  


## Further Validate the Optimum Number of NMF Components

- To verify that 3 components is the Optimum number, multiple NMF models where fitted for a range of number of components. The euclidean norm of the residuals was calculated for each model. 

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/4a88e94c-dbf3-408b-8c89-42eae04794de)



**Insights:** 

- The curve shows an inflection point at 3 components. This point represents the best compromise between residuals and number of components. 

- The result found by hierarchical clustering is validated.


## NMF Model Evaluation: Validate Reconstruction

- An intermediate data point (at the nucleation) is used to evaluate the quality of the NMF model to reconstruct the original data.
 
- The original and reconstructed spectra are plotted and compared, as well as the residuals. 

![image](https://github.com/solutioncrafter/Finding_Chemical_Signatures_with_Usupervised_Learning/assets/126869447/9af9aba3-adad-4cc7-938a-a7d6e0c5732e)


**Insights:** 
- There is a match between the reconstructed and original spectra. 

- The residuals have no defined structure and a magnitude below 5%. 


# 9. Conclusions 


- By applying hierarchical clustering and a series of scores, the optimum number of chemical signatures from the dataset was determined to be 3. The result was further validated by visual inspection on a 2-dimensional reduced representation of de data.
- 3 components were extracted from the dataset by NMF which are the primary chemical signatures describing the chemical species involved in the crystallization process.
- The reduced representation obtained from NMF (weights) revealed insights about the behavior of each component throughout the crystallization process.

- Both, the extracted components and weights are highly interpretable results. 

- Although this notebook is focused on the data analysis, the role of a domain expert plays a key role in the interpretation of the results, as it was described in the original paper.


# 10. References

- Urquidi, Oscar, et al. "In situ optical spectroscopy of crystallization: One crystal nucleation at a time." Proceedings of the National Academy of Sciences 119.16 (2022): e2122990119.

- Data source: https://doi.org/10.1073/pnas.2122990119)


