FROM bioconductor/tidyverse:3.16


RUN Rscript -e 'BiocManager::install(c("pheatmap", "viridis", \
                       "zoo", "devtools", "tiff", \
                       "distill", "openxlsx", "ggrepel", "patchwork", "mclust", \
                       "RColorBrewer", "uwot", "Rtsne", "cowplot", "kohonen", "caret", \ 
                       "randomForest", "ggridges", "cowplot", "gridGraphics", \
                       "scales", "CATALYST", "scuttle", "scater", \
                       "dittoSeq", "tidyverse", "batchelor", \
                       "bluster","scran", "lisaClust", "spicyR", "imcRtools", \
                       "cytomapper", "imcdatasets"))'
RUN Rscript -e 'devtools::install_github("i-cyto/Rphenograph")'
RUN Rscript -e 'BiocManager::install("harmony")'
RUN Rscript -e 'BiocManager::install("zellkonverter")'
RUN Rscript -e 'devtools::install_github("stuchly/Rphenoannoy")'
RUN Rscript -e 'devtools::install_github("BodenmillerGroup/cytoviewer")'
