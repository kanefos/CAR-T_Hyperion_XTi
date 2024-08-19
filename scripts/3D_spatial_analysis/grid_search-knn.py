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
import spatial_analysis.spatial_analysis as sa
from td_graph.td_graph import Neighbors
import warnings
warnings.filterwarnings("ignore")


# Load the data
adata = sc.read('samples/adata_final.h5ad')
adata=adata[~adata.obs['major_pixel_type'].isin(['Unclassified','Low markers'])]
spatial=adata.obs[['x','y','z']].to_numpy()
adata.obsm['spatial']=spatial
# Set the parameters for the grid search

radius=[4,6,8,10,15,26,30]
n_clusters=[6,8,10,12,14,16,18,20,25]


count_by= "major_pixel_type"
raw_cols=['library_id','image',
           'major_pixel_type',
           'functional_pixel_type',
           'cn_celltypes',
           'SpatialContext']
celltypes={
    'major':'major_pixel_type',
    'functional':'functional_pixel_type'
}

graph='knn'

def compute_graph(adata,n_neighbors):
    adata = Neighbors.spatial_neighbors(adata, library_key='image', graph_type='knn',n_neighbors=n_neighbors)
    return adata

def compute_spatial_features(adata,n_clusters):
    # aggregate neighbours
    adata=sa.Neighbors.aggregateNeighbors(adata=adata,
                    connectivity_key='3d_connectivities',
                    aggregate_by= 'metadata',
                    count_by= 'major_pixel_type',
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
                 connectivity_key='3d_connectivities',
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


def get_pixels(adata,cluster:str, id_key: str ='DFCI_id'):
    areas=adata.obs[[id_key,cluster]].value_counts()
    areas=areas.reset_index(name='pixels')
    total_areas=areas.groupby(id_key)['pixels'].sum()
    total_areas=total_areas.reset_index(name='total_pixels')
    areas=areas.merge(total_areas,on=id_key)
    areas['proportion']=areas['pixels']/areas['total_pixels']
    return areas

def extract_results(adata,id_key='image',raw_cols=raw_cols):
    cn_areas=get_pixels(adata=adata,cluster='cn_celltypes', id_key=id_key)
    sc_areas=get_pixels(adata=adata,cluster='SpatialContext', id_key=id_key)
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
                    graph='knn',
                    outdir='./results'):
    for n_neighbors in radius:
        adata=compute_graph(adata,n_neighbors)
        for cn in n_clusters:
            adata=compute_spatial_features(adata=adata,n_clusters=cn)
            interactions_dict=compute_interactions(adata=adata,celltypes=celltypes,connectivity_key='3d')
            cn_areas,sc_areas,cn_props,sc_props,raw_results=extract_results(adata=adata,id_key='image',raw_cols=raw_cols)
            save_results(cn_areas=cn_areas,
                sc_areas=sc_areas,
                cn_props=cn_props,
                sc_props=sc_props,
                raw_results=raw_results,
                interactions_dict=interactions_dict,
                k=n_neighbors, 
                graph=graph,
                n_clusters=cn,
                dataset=dataset,
                outdir=outdir)
            gc.collect()

############################################################################################################################################################################
# 3D
############################################################################################################################################################################
dataset='3d'



perform_analysis(adata=adata,
                    radius=radius,
                    n_clusters=n_clusters,
                    count_by=count_by,
                    raw_cols=raw_cols,
                    celltypes=celltypes,
                    dataset=dataset,
                    graph='knn',
                    outdir='./results')