#!/usr/bin/bash

steinbock() {
    docker run -v "/mnt/disks/kenobi/yoshi/ImagingWorkshop2023/data/non_denoised":/data -u $(id -u):$(id -g) ghcr.io/bodenmillergroup/steinbock:0.15.0 "$@"
}

steinbock --version

steinbock segment deepcell --app mesmer --minmax

rm -f data/non_denoised/masks/*
cp data/mydata/masks/* data/non_denoised/masks/
steinbock measure intensities --aggr mean
steinbock measure regionprops
steinbock measure neighbors --type expansion --dmax 4
# steinbock export ome
# steinbock export histocat
steinbock export csv intensities regionprops -o cells.csv
# steinbock export fcs --no-concat intensities regionprops -o fcs
steinbock export anndata --no-concat --intensities intensities --data regionprops --neighbors neighbors -o anndata
steinbock export graphs --format graphml --data intensities
