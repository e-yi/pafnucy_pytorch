##pafnucy_pytoch
a pytorch implementation of Pafnucy.

Please find more about Pafnucy [here](https://gitlab.com/cheminfIBB/pafnucy).

-----
**Pafnucy** is a 3D convolutional neural network that predicts binding affinity for protein-ligand complexes.
It was trained on the [PDBbind](http://pubs.acs.org/doi/abs/10.1021/acs.accounts.6b00491) database and tested on the [CASF](http://pubs.acs.org/doi/pdf/10.1021/ci500081m) "scoring power" benchmark.

The manuscript describing Pafnucy was published in *Bioinformatics* [DOI: 10.1093/bioinformatics/bty374](https://doi.org/10.1093/bioinformatics/bty374).

### Requirements
* Python 3.6+
* openbabel 2.4  [openbabel 3 has changed its API]
* numpy 
* h5py 
* pytorch
* matplotlib
* scikit-learn 
* scipy

---
`python main.py -i path/to/data`

Data is provided in [the original repo](https://gitlab.com/cheminfIBB/pafnucy).