############################################################################################################################################################################
# This script is used to run the grid search and analyze each combination of parameters (k and CNs) for the spatial analysis.
############################################################################################################################################################################
# Importing the necessary libraries

import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import pickle
import gc
import spatial_analysis.spatial_analysis as sa
import BorderGraph.BorderGraph as bg
import warnings
warnings.filterwarnings("ignore")


# Load the data
adata = sc.read('data/non_denoised/spe_final_adipocytes.h5ad')

adata=adata[~adata.obs['minor_cell_type'].isin(['B-T','Unclassified','Artifact'])]
adata=adata[~adata.obs['medium_cell_type'].isin(['B-T','Unclassified','Artifact','Other T'])]
adata=adata[~adata.obs['major_cell_type'].isin(['B-T','Unclassified','Artifact'])]
adata=adata[adata.obs_names != 's40905_2708']

# subset to test the script
# libraries=adata.obs['library_id'].cat.categories
# adata=adata[adata.obs['library_id'].isin(libraries[:2])]
# adata.uns['spatial']={k:v for k,v in adata.uns['spatial'].items() if k in adata.obs['library_id'].cat.categories}
# Set the parameters for the grid search

radius=[5,10,30]
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

connectivity_key='border'


# Compute the contours
# adata=bg.Contours.contours_per_image(adata)

contours_file=open('contours.pkl','rb')
contours=pickle.load(contours_file)
adata.uns['contours']=contours
del contours
gc.collect()

def compute_graph(adata,cutoff,key_added='border'):
    results = bg.Distances.compute_all_images(adata, cutoff=cutoff)
    adata = bg.Utils.update_adata(adata,key_added, results, cutoff)
    return adata



def compute_spatial_features(adata,n_clusters):
    # aggregate neighbours
    adata=sa.Neighbors.aggregateNeighbors(adata=adata,
                    connectivity_key='border_connectivities',
                    aggregate_by= 'metadata',
                    count_by= 'medium_cell_type',
                    proportions = True,
                    layer='exprs')   
    df=adata.uns['aggregatedNeighbors']
    # compute clusters
    clusters,y=sa.CN.compute_cell_neigh(adata=adata,
                                    n_clusters=n_clusters,
                                    key_connectivity='aggregatedNeighbors')
    adata=sa.utils.incorporate_clusters(adata=adata, clusters=clusters, y=y, column='cn_celltypes')
    # aggregate neighbours CNs
    adata=sa.Neighbors.aggregateNeighbors(adata=adata,
                 connectivity_key='border_connectivities',
                 aggregate_by= 'metadata',
                 count_by= 'cn_celltypes',name='aggregatedCN',
                 proportions = True)
    #compute SCs
    adata=sa.SC.detectSpatialContext(adata=adata,
                        neighbors_key='aggregatedCN',
                                name='SpatialContext')
    return adata


def compute_interactions(adata,celltypes,connectivity_key):
    interactions_dict=sa.Interactions.get_interaction_dict(adata=adata,
                                                        column='cn_celltypes',
                                                        connectivity_key=connectivity_key,
                                                        celltypes=celltypes)
    interactions_dict=sa.utils.prepare_interaction_df(interactions_dict,'cn_celltypes')
    return interactions_dict




def extract_results(adata,id_key='DFCI_id',raw_cols=raw_cols):
    cn_areas=sa.utils.get_areas(adata=adata,cluster='cn_celltypes', id_key=id_key)
    sc_areas=sa.utils.get_areas(adata=adata,cluster='SpatialContext', id_key=id_key)
    cn_props={}
    sc_props={}
    for key,value in celltypes.items():
        cn_props[key]=sa.utils.get_cluster_props(adata=adata,cluster='cn_celltypes',celltype=value, id_key=id_key)
        sc_props[key]=sa.utils.get_cluster_props(adata=adata,cluster='SpatialContext',celltype=value, id_key=id_key)
    raw_results=adata.obs[raw_cols]
    return cn_areas,sc_areas,cn_props,sc_props,raw_results

def save_results(cn_areas,sc_areas,cn_props,sc_props,raw_results,interactions_dict,
                k, graph, n_clusters,dataset, outdir):
    sa.utils.save_tables(df=cn_areas,
        graph=graph, 
        k=k, 
        n_clusters=n_clusters, 
        dataset=dataset, 
        outdir=outdir, 
        datatype='CN', 
        metrics='area')
    sa.utils.save_tables(df=sc_areas,
        graph=graph, 
        k=k, 
        n_clusters=n_clusters, 
        dataset=dataset, 
        outdir=outdir, 
        datatype='SC', 
        metrics='area') 
    sa.utils.save_tables(df=raw_results,
        graph=graph, 
        k=k, 
        n_clusters=n_clusters, 
        dataset=dataset, 
        datatype='raw-results', 
        outdir=outdir)
    for key,value in cn_props.items():
        sa.utils.save_tables(df=value,
            graph=graph,
            k=k,
            n_clusters=n_clusters,
            dataset=dataset,
            datatype='CN',
            metrics='proportions',
            celltype=key,
            outdir=outdir)
    for key,value in sc_props.items():
        sa.utils.save_tables(df=value,
            graph=graph,
            k=k,
            n_clusters=n_clusters,
            dataset=dataset,
            datatype='SC',
            metrics='proportions',
            celltype=key,
            outdir=outdir)
    for key,value in interactions_dict.items():
        sa.utils.save_tables(df=value,
            graph=graph,
            k=k,
            n_clusters=n_clusters,
            dataset=dataset,
            datatype='interactions',
            celltype=key,
            outdir=outdir)




def perform_analysis(adata,
                    radius,
                    n_clusters,
                    count_by,
                    raw_cols,
                    celltypes,
                    dataset,
                    graph='border',
                    outdir='./results',
                    key_added='border'):
    for cutoff in radius:
        results = bg.Distances.compute_all_images(adata, cutoff=cutoff)
        adata = bg.Utils.update_adata(adata,key_added, results, cutoff)
        for cn in n_clusters:
            adata=compute_spatial_features(adata=adata,n_clusters=cn)
            interactions_dict=compute_interactions(adata=adata,celltypes=celltypes,connectivity_key=connectivity_key)
            cn_areas,sc_areas,cn_props,sc_props,raw_results=extract_results(adata=adata,id_key='DFCI_id',raw_cols=raw_cols)
            save_results(cn_areas=cn_areas,
                sc_areas=sc_areas,
                cn_props=cn_props,
                sc_props=sc_props,
                raw_results=raw_results,
                interactions_dict=interactions_dict,
                k=cutoff, 
                graph=graph,
                n_clusters=cn,
                dataset=dataset,
                outdir=outdir)
            gc.collect()

############################################################################################################################################################################
# Baseline
############################################################################################################################################################################
dataset='baseline'

adata_tmp=adata[adata.obs['timepoint'] == 'Baseline'].copy()
# Remove not needed images to light the memory load
adata_tmp.uns['spatial']={k:v for k,v in adata_tmp.uns['spatial'].items() if k in adata_tmp.obs['library_id'].unique()}
adata_tmp.uns['contours']={k:v for k,v in adata_tmp.uns['contours'].items() if k in adata_tmp.obs['library_id'].unique()}

# perform_analysis(adata=adata_tmp,
#                     radius=radius,
#                     n_clusters=n_clusters,
#                     count_by=count_by,
#                     raw_cols=raw_cols,
#                     celltypes=celltypes,
#                     dataset=dataset,
#                     graph='border',
#                     outdir='./results')




############################################################################################################################################################################
# No myeloma
############################################################################################################################################################################
dataset='no_myeloma'

# Remove Myeloma cells from baseline
adata_tmp=adata_tmp[~adata_tmp.obs['minor_cell_type'].isin(['Myeloma'])]
adata_tmp.uns['contours']={k:v for k,v in adata_tmp.uns['contours'].items() if k in adata_tmp.obs['library_id'].unique()}

perform_analysis(adata=adata_tmp,
                    radius=radius,
                    n_clusters=n_clusters,
                    count_by=count_by,
                    raw_cols=raw_cols,
                    celltypes=celltypes,
                    dataset=dataset,
                    graph='border',
                    outdir='./results')

del adata_tmp
gc.collect()

############################################################################################################################################################################
# Post-treatment
############################################################################################################################################################################
dataset='01MOpost'

# select post threatment samples
adata_tmp=adata[adata.obs['timepoint'] == '01MOpost'].copy()
# Remove not needed images to light the memory load
adata_tmp.uns['spatial']={k:v for k,v in adata_tmp.uns['spatial'].items() if k in adata_tmp.obs['library_id'].unique()}
adata_tmp.uns['contours']={k:v for k,v in adata_tmp.uns['contours'].items() if k in adata_tmp.obs['library_id'].unique()}


perform_analysis(adata=adata_tmp,
                    radius=radius,
                    n_clusters=n_clusters,
                    count_by=count_by,
                    raw_cols=raw_cols,
                    celltypes=celltypes,
                    dataset=dataset,
                    graph='border',
                    outdir='./results')


del adata_tmp
gc.collect()

############################################################################################################################################################################
# Paired samples
############################################################################################################################################################################

dataset='baseline_post'

ids=adata.obs[['timepoint','DFCI_id']]
ids=ids.groupby('DFCI_id').filter(lambda x: set(x['timepoint']) == {'Baseline', '01MOpost'}).drop_duplicates().sort_values(by='DFCI_id')
adata_tmp=adata[adata.obs['DFCI_id'].isin(ids['DFCI_id'].unique())].copy()
# Remove not needed images to light the memory load
adata_tmp.uns['spatial']={k:v for k,v in adata_tmp.uns['spatial'].items() if k in adata_tmp.obs['library_id'].unique()}
adata_tmp.uns['contours']={k:v for k,v in adata_tmp.uns['contours'].items() if k in adata_tmp.obs['library_id'].unique()}

perform_analysis(adata=adata_tmp,
                    radius=radius,
                    n_clusters=n_clusters,
                    count_by=count_by,
                    raw_cols=raw_cols,
                    celltypes=celltypes,
                    dataset=dataset,
                    graph='border',
                    outdir='./results')

del adata_tmp
gc.collect()