############################################################################################################################################################################
# This script is used to run the grid search and analyze each combination of parameters (k and CNs) for the spatial analysis.
############################################################################################################################################################################
# Importing the necessary libraries

import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import gc
from spatial_analysis import spatial_analysis
import warnings
warnings.filterwarnings("ignore")


# Load the data
adata = sc.read('data/non_denoised/spe_final_adipocytes.h5ad')

adata=adata[~adata.obs['minor_cell_type'].isin(['B-T','Unclassified','Artifact'])]
adata=adata[~adata.obs['medium_cell_type'].isin(['B-T','Unclassified','Artifact','Other T'])]
adata=adata[~adata.obs['major_cell_type'].isin(['B-T','Unclassified','Artifact'])]

# Set the parameters for the grid search

radius=[4,6,8,10,15,30]
n_clusters=[6,8,10,12,14,16,18,20,25]


count_by= "medium_cell_type"
raw_cols=['library_id','timepoint','DFCI_id',
           'major_cell_type','medium_cell_type','minor_cell_type','functional_minor_cell_type',
           'area',
           'cn_celltypes',
           'SpatialContext']
celltypes={
    'functional':'functional_minor_cell_type',
    'minor':'minor_cell_type',
    'medium':'medium_cell_type',
    'major':'major_cell_type'
}

graph='knn'

## define loop to run the spatial analysis on the parameters paris
def grid_search(adata:AnnData, 
                radius:list[float], 
                n_clusters:list[int],
                graph:str,
                dataset:str,
                count_by:str,
                raw_cols=list[str],
                celltypes=dict[str:str])->None:

    for rad in radius:
        for cn in n_clusters:
            spatial_analysis.runSpatialAnalysis(adata=adata_tmp,
                                        k=rad,
                                        n_clusters=cn,
                                        graph=graph,
                                        dataset=dataset,
                                        count_by=count_by,
                                        raw_cols=raw_cols,
                                        celltypes=celltypes)
    return None
############################################################################################################################################################################
# Baseline
############################################################################################################################################################################
dataset='baseline'

adata_tmp=adata[adata.obs['timepoint'] == 'Baseline'].copy()
# Remove not needed images to light the memory load
adata_tmp.uns['spatial']={k:v for k,v in adata_tmp.uns['spatial'].items() if k in adata_tmp.obs['library_id']}

grid_search(adata=adata_tmp,
            radius=radius, 
            n_clusters=n_clusters, 
            graph=graph, 
            dataset=dataset,
            count_by=count_by, 
            raw_cols=raw_cols, 
            celltypes=celltypes)


############################################################################################################################################################################
# No myeloma
############################################################################################################################################################################
dataset='no_myeloma'

# Remove Myeloma cells from baseline
adata_tmp=adata_tmp[~adata_tmp.obs['minor_cell_type'].isin(['Myeloma'])]

grid_search(adata=adata_tmp,
            radius=radius, 
            n_clusters=n_clusters, 
            graph=graph, 
            dataset=dataset,
            count_by=count_by, 
            raw_cols=raw_cols, 
            celltypes=celltypes)

del adata_tmp
gc.collect()

# ############################################################################################################################################################################
# # Post-treatment
# ############################################################################################################################################################################
# dataset='01MOpost'

# # select post threatment samples
# adata_tmp=adata[adata.obs['timepoint'] == '01MOpost'].copy()
# # Remove not needed images to light the memory load
# adata_tmp.uns['spatial']={k:v for k,v in adata_tmp.uns['spatial'].items() if k in adata_tmp.obs['library_id']}


# grid_search(adata=adata_tmp,
#             radius=radius, 
#             n_clusters=n_clusters, 
#             graph=graph, 
#             dataset=dataset,
#             count_by=count_by, 
#             raw_cols=raw_cols, 
#             celltypes=celltypes)

# del adata_tmp
# gc.collect()

# ############################################################################################################################################################################
# # Paired samples
# ############################################################################################################################################################################

# dataset='baseline_post'

# ids=adata.obs[['timepoint','DFCI_id']]
# ids=ids.groupby('DFCI_id').filter(lambda x: set(x['timepoint']) == {'Baseline', '01MOpost'}).drop_duplicates().sort_values(by='DFCI_id')
# adata_tmp=adata[adata.obs['DFCI_id'].isin(ids['DFCI_id'].unique())].copy()
# # Remove not needed images to light the memory load
# adata_tmp.uns['spatial']={k:v for k,v in adata_tmp.uns['spatial'].items() if k in adata_tmp.obs['library_id']}

# grid_search(adata=adata_tmp,
#             radius=radius, 
#             n_clusters=n_clusters, 
#             graph=graph, 
#             dataset=dataset,
#             count_by=count_by, 
#             raw_cols=raw_cols, 
#             celltypes=celltypes)

# del adata_tmp
# gc.collect()

