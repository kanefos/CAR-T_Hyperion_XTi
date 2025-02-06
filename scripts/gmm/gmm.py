import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time
import gc
from sklearn.mixture import GaussianMixture

class GMMCluster():
    @staticmethod
    def search_bic(mat:np.ndarray,n_components_range:list[int],covariance_types:list[str], random_state:int=12345,max_iter:int=99999)->dict:
        """
        Optimize the number of components and the covariance type of a Gaussian Mixture Model using the Bayesian Information Criterion.
        """
        np.random.seed(random_state)
        bic_scores = {}#
        for covariance_type in covariance_types:
            bic_scores[covariance_type] = []
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state,max_iter=max_iter)
                gmm.fit(mat)
                bic = gmm.bic(mat)
                bic_scores[covariance_type].append(bic)
        return bic_scores

    
    @staticmethod
    def get_best_params(bic_scores:dict,covariance_types:list[int],n_components_range:list[str])->dict:
        """
        Get the best number of components and covariance type from the BIC scores.
        """
        best_covariance_type = None
        best_n_components = None
        min_bic = np.inf
        for covariance_type in covariance_types:
            min_bic_covariance = min(bic_scores[covariance_type])
            if min_bic_covariance < min_bic:
                min_bic = min_bic_covariance
                best_covariance_type = covariance_type
                best_n_components = n_components_range[np.argmin(bic_scores[covariance_type])]
        best_params = {'covariance_type':best_covariance_type,'n_components':best_n_components}
        return best_params
    
    @staticmethod
    def train_best_gmm(mat:np.ndarray,best_params:dict,random_state:int=12345,max_iter:int=99999)->GaussianMixture:
        """
        Train a Gaussian Mixture Model with the best number of components and covariance type.
        """
        gmm = GaussianMixture(n_components=best_params['n_components'], covariance_type=best_params['covariance_type'], random_state=random_state,max_iter=max_iter)
        gmm.fit(mat)
        return gmm

    @staticmethod
    def plot_bic_scores(bic_scores:dict,n_components_range:list[int],covariance_types:list[str])->None:
        """
        Plot the BIC scores for each number of components and covariance type.
        """
        plt.figure(figsize=(10, 8))
        for covariance_type in covariance_types:
            plt.plot(n_components_range, bic_scores[covariance_type], label=covariance_type)
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC score')
        plt.title('BIC Scores for Different Numbers of Clusters and Covariance Types')
        plt.legend()
        plt.show()

    @classmethod
    def get_best_gmm(cls,
                    mat:np.ndarray,
                    n_components_range:list[int],
                    covariance_types:list[str],
                    random_state=12345, 
                    max_iter=99999, 
                    plot:bool=False)->GaussianMixture:
        """
        Get the best Gaussian Mixture Model from the BIC scores.
        """
        # search BIC scores
        bic_scores = cls.search_bic(mat, n_components_range, covariance_types, random_state, max_iter)
        # get best parameters
        best_params = cls.get_best_params(bic_scores, covariance_types,n_components_range)
        # train best GMM
        gmm = cls.train_best_gmm(mat, best_params, random_state, max_iter)
        if plot:
            cls.plot_bic_scores(bic_scores, n_components_range, covariance_types)
        return gmm