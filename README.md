
# InfoTAD: Applying Infomap Graph Partitioning to Hi-C Contact Map


The InfoTAD method for TAD identification based on Infomap entropy.
## Requirements

python>=3.7

pandas

numpy


## Quick Start

We offer the example of running simulated data in example.ipynb

Additionally, for real Hi-C matrix, you can run command as follow:
```
cd InfoTAD

#run example real Hi-C matrix in InfoTAD/example_data/

python run_real_hic.py -i example_data/chr19_25kb_99_200.txt -o ./ -s 2500000 -c chr19 -r 25000 -step 1
```
## Input and Output 

The input should be symmetric Hi-C contact matrix. Example matrices are in the folder "example_data".

For simulated data, the output is a two-line txt file with the left and right boundaries. All the left boundaries are 
in the first line, and right ones in the second line.

For real Hi-C data, we output both two-line txt and eight columns .tsv files. 
An example output of the tsv file is shown below (resolution=1kb):
```
chr1	1	0       1000	chr1	44	43000	44000

chr1	9	8000	9000	chr1	16	15000	16000

chr1	17	16000	17000	chr1	44	43000	44000
```

## Generate simulated data

In folder "example_data", except real Hi-C example, we also offer a simulated matrix and its ground truth label txt. 
You can generate the simulated matrices by command like
```
cd InfoTAD/scripts

python generate_sim_batch.py -N 0.1 -M 10 -m 8 -S 10 -s 8 -e 0.7 -T 1 -D ./example_data/
```
### parameters 

````
-N noise_ratio
-M maximum number of vertices
-m minimum number of vertices
-S maximum TAD size
-s minimum TAD size
-e probability of intra-interaction 
-T number of matrices
-D output directory
````

## Contact

Feel free to open an issue in Github or contact liangqs@tongji.edu.cn if you have any problem in using InfoTAD.