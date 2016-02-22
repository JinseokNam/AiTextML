Implementation of the following paper

```
@INPROCEEDINGS{
  author = {Jinseok Nam, Eneldo Loza Menc{\'i}a, Johannes F{\"u}rnkranz},
  title = {All-in Text: Learning Document, Label, and Word Representations Jointly},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2016}
}
```


# Install external libraries

Required external libraries
- gflags
- glog
- OpenBLAS
- Boost C++
- HDF5

### OpenBLAS

```
git clone -b 'v0.2.15' https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make
make PREFIX=${HOME}/local install
```

### gflags

```
git clone -b 'v2.1.2' https://github.com/gflags/gflags.git
cd gflags
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/local ..
make
make install
```

Make sure that `cmake` is installed on your system.

### glog

```
git clone -b 'v0.3.4' https://github.com/google/glog.git
cd glog
./configure --prefix=${HOME}/local && make && make install
```

If you have an error related to `aclocal` while installing `glog`, please install `automake1.4`

On Ubuntu 15.04, `automake1.4` can be installed by using the following command.

```
sudo apt-get install automake1.4
```


### Boost C++

```
wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
tar xvzf boost_1_58_0.tar.gz && cd boost_1_58_0
./bootstrap.sh --prefix=${HOME}/local
./b2 install
```

### HDF5

```
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar
tar xvf hdf5-1.8.16.tar && cd hdf5-1.8.16
./configure --prefix=${HOME}/local --enable-threadsafe --enable-cxx --enable-unsupported
make && make install
```

# BioASQ dataset and 2015 MeSH

The BioASQ dataset used in the paper is available at the following link.

http://participants-area.bioasq.org/Tasks/3a/trainingDataset/raw/allMeSH/

In order to download the data file, you need to log in BioASQ.
For more information, please visit http://participants-area.bioasq.org

You also need a file of MeSH descriptors in XML format.

http://www.nlm.nih.gov/mesh/filelist.html

Please note that we used 2015 MeSH in our experiments.

# Preprocessing
Once the raw dataset (allMeSH.json after uncompressed) is downloaded from BioASQ, you can create a dataset file with the following command.

```
export LD_LIBRARY_PATH=${HOME}/local/lib:${LD_LIBRARY_PATH}
./preproc/data_prep_pipeline.sh <path/to/BioASQ_json_file> <path/to/MeSH2015_xml> <output_directory>
```

The script runs preprocessing scripts such as extraction of MeSH descriptors from XML file and tokenization, splits train and test documents by year, and then creates a HDF5 file which contains all necessary information to train our models.

Preprocessing takes several hours to complete and creates multiple text files.

All the information for experiments can be found in the HDF5 file (dataset.h5).

# Train the model

Let's assume that the dataset file generated from the preprocessing step is stored under `data/BioASQ_preprocessed` and we want to save the model parameters to `models/BioASQ_model`.
You can also specify the number of threads `--num_threads` to be used for parameter updates in the training course.

```
bin/aitextml --mode train --dataset data/BioASQ_preprocessed/dataset.h5 --num_iters 10 --save_train_model models/BioASQ_model --num_threads 8
```
