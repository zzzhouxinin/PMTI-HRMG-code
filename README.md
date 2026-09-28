# PMTI-HRMG

**Plant miRNA-Target Interaction Prediction via Hierarchical Multi-scale Graph Neural Network**

An end-to-end plant miRNA-target interaction prediction framework based on hierarchical multi-scale graph neural networks.

## Project Overview

PMTI-HRMG is a plant miRNA and target gene interaction prediction pipeline. The method constructs a gene-miRNA heterogeneous graph, fuses sequence features with topological features, and performs link prediction using graph neural networks (GNNs) to identify potential miRNA-target gene regulatory relationships.

### Key Features

- **Heterogeneous graph representation**: Models gene-gene, miRNA-miRNA, and gene-miRNA relationships simultaneously
- **Multi-feature fusion**: Combines sequence K-mer features (DAE encoded) with graph topological features (SVD encoded)
- **Multi-scale graph convolution**: Supports multiple graph convolution methods: GCN, ChebConv, TAGConv, and GAT
- **Hierarchical attention mechanism**: The HRMG model automatically learns the importance of different scales via multi-scale attention
- **K-fold cross validation**: Provides a rigorous model evaluation scheme (default 5-fold)

## Directory Structure

```
PMTI-HRMG/
├── Dataset.zip                     # Raw dataset archive (extract into data/ directory)
├── data/                          # Raw data directory (created after extraction)
│   ├── Arabidopsis_thaliana/      # Arabidopsis thaliana
│   ├── Prunus_persica/            # Peach
│   └── Solanum_lycopersicum/      # Tomato
│       ├── PmiREN/               # miRNA data
│       │   ├── *_mature.fa
│       │   └── *_targetGene.txt
│       └── Genome/               # Genome data
│           └── *.transcript.fa / *.primaryTrs.fa / *.cdna.all.fa
├── process_data/                  # Intermediate processed data (auto-generated)
├── train_result/                  # Training results (auto-generated)
├── picture_results/               # Visualization results (auto-generated)
├── model/                         # Model definitions
│   ├── HRMG.py                   # Hierarchical multi-scale graph neural network (core model)
│   ├── GraphConvolutional.py     # GCN model
│   ├── ChebConvConvolutional.py  # ChebConv model
│   ├── GAT.py                    # Graph attention network
│   ├── DAE.py                    # Denoising autoencoder
│   ├── MLP.py                    # Multi-layer perceptron (link prediction head)
│   ├── MakePygFormat.py          # PyG data format conversion
│   └── PygLinkPrediction.py      # Link prediction wrapper
├── graphconstruct.py              # Step 1: Heterogeneous graph construction
├── feture_encoded.py              # Step 2: Feature encoding and fusion
├── model_train.py                 # Step 3: Model training and evaluation
├── utils.py                       # Utility functions
└── README.md
```

## Requirements

### Hardware Requirements

- NVIDIA GPU (recommended, CUDA enabled) or CPU
- Memory: 16GB+ recommended

### Software Dependencies

```
Python >= 3.8
PyTorch >= 1.10.0
PyTorch Geometric >= 2.0.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib (optional, for visualization)
psutil (optional, for memory monitoring)
```

### Installation

```bash
# Create virtual environment
conda create -n pmt python=3.8
conda activate pmt

# Install PyTorch (choose according to your CUDA version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install other dependencies
pip install pandas numpy scikit-learn matplotlib psutil
```

## Usage

### Data Preparation

A `Dataset.zip` archive is provided in the project root directory, containing complete test data for three species. Simply extract it into the current directory:

```bash
# Windows PowerShell
Expand-Archive -Path Dataset.zip -DestinationPath .\data -Force

# Or use Python
python -c "import zipfile; zipfile.ZipFile('Dataset.zip').extractall('data')"
```

After extraction, the `data/` directory contains the following species datasets:

| Species | Directory Name | Transcript File |
|---------|---------------|-----------------|
| Arabidopsis thaliana | `Arabidopsis_thaliana` | `Athaliana_167_TAIR10.transcript.fa` |
| Peach | `Prunus_persica` | `Prunus_persica_v2.0.a1.primaryTrs.fa` |
| Tomato | `Solanum_lycopersicum` | `Solanum_lycopersicum.SL3.0.cdna.all.fa` |

Structure of each species directory:

```
data/<species>/
├── PmiREN/
│   ├── <species>_mature.fa          # Mature miRNA sequences
│   ├── <species>_targetGene.txt     # miRNA-target gene pairs
│   ├── <species>_hairpin.fa          # miRNA hairpin sequences
│   ├── <species>_basicInfo.txt       # Basic miRNA information
│   └── ...
└── Genome/
    └── <species>_*.fa                 # Transcript sequences
```

### Step 1: Construct Heterogeneous Graph

Run [graphconstruct.py](file:///c:/Users/ZhuanZ/Desktop/PMTI-HRMG/graphconstruct.py) to build the gene-miRNA heterogeneous graph:

```bash
python graphconstruct.py \
    --prefix Arabidopsis_thaliana \
    --miRNA_fa ./data/Arabidopsis_thaliana/PmiREN/Arabidopsis_thaliana_mature.fa \
    --target ./data/Arabidopsis_thaliana/PmiREN/Arabidopsis_thaliana_targetGene.txt \
    --transcript ./data/Arabidopsis_thaliana/Genome/Athaliana_167_TAIR10.transcript.fa \
    --kmer 7
```

**Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prefix` | Arabidopsis_thaliana | Output file prefix (use species name recommended) |
| `--miRNA_fa` | - | Path to mature miRNA sequence fasta file |
| `--target` | - | Path to miRNA-target gene pairs file |
| `--transcript` | - | Path to transcript sequence fasta file |
| `--kmer` | 7 | K-mer length |

This step generates the following files in the `process_data/` directory:

- `*_kmer.csv`: K-mer frequency matrix
- `*_cosine_sim.csv` / `*_jaccard_sim.csv`: Sequence similarity matrices
- `*_G_G_*_graph.csv`: Gene-gene subgraph
- `*_M_M_*_graph.csv`: miRNA-miRNA subgraph
- `*_G_M_*_graph.csv`: Gene-miRNA subgraph
- `*_M_G_*_graph.csv`: miRNA-gene subgraph
- `*_*_Heterogeneous_graph.csv`: Full heterogeneous graph
- `*_node_attr.csv`: Node attribute matrix

### Step 2: Feature Encoding and Fusion

Run [feture_encoded.py](file:///c:/Users/ZhuanZ/Desktop/PMTI-HRMG/feture_encoded.py) to encode node features:

```bash
python feture_encoded.py \
    --prefix Arabidopsis_thaliana \
    --device cuda:0 \
    --lr 0.001 \
    --epoch 1000 \
    --D 256 \
    --graph_dim 64 \
    --batch_size 512
```

**Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prefix` | Arabidopsis_thaliana | Output file prefix (must match Step 1) |
| `--device` | cuda:0 | Compute device (cuda/cpu) |
| `--lr` | 0.001 | DAE learning rate |
| `--epoch` | 1000 | DAE training epochs |
| `--D` | 256 | Sequence feature embedding dimension (try 16/32/64/128/256/512/1024) |
| `--graph_dim` | 64 | Topological feature embedding dimension (SVD) |
| `--batch_size` | 512 | Batch size |

This step outputs:

- `*_node_attr_encode.csv`: Fused node feature matrix

### Step 3: Model Training and Evaluation

Run [model_train.py](file:///c:/Users/ZhuanZ/Desktop/PMTI-HRMG/model_train.py) to train the prediction model:

```bash
python model_train.py \
    --method HRMG \
    --prefix Arabidopsis_thaliana \
    --device cuda:0 \
    --lr 0.0001 \
    --epoch 3000 \
    --kfold 5 \
    --threshold 0.5 \
    --graphdim1 2048 \
    --graphdim2 1024 \
    --graphdim3 512 \
    --mlpdim2 256 \
    --mlpdim3 64 \
    --simthreshold 0.4 \
    --D 256 \
    --sim jaccard
```

**Key Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | GCN | Graph convolution method: 'HRMG','GCN','ChebConv','GAT'
| `--prefix` | Prunus_persica | File prefix (must match Steps 1 and 2) |
| `--device` | cpu | Compute device |
| `--lr` | 0.0001 | Learning rate |
| `--epoch` | 3000 | Training epochs |
| `--kfold` | 5 | Number of cross-validation folds |
| `--threshold` | 0.5 | Classification threshold |
| `--graphdim1/2/3` | 2048/1024/512 | Dimensions of each graph convolution layer |
| `--mlpdim2/3` | 256/64 | MLP hidden layer dimensions |
| `--simthreshold` | 0.4 | Graph edge similarity threshold |
| `--D` | 256 | Node feature dimension (must match Step 2) |
| `--sim` | cosine | Similarity method: `cosine` / `jaccard` |
| `--chebconvk` | 1 | K value for ChebConv |
| `--heads` | 1 | Number of GAT attention heads |

Training results are saved in the `train_result/` directory, including:

- `*_*_K_*_scores.csv`: Prediction scores
- `*_*_K_*_fpr/tpr.csv`: ROC curve data
- `*_*_K_*_precision/recall.csv`: PR curve data
- `*_*_K_*_loss.csv`: Training loss
- `*_*_K_*_GRAPH/MLP.pt`: Model weights

## Model Architecture

### HRMG (Hierarchical Multi-scale Graph Neural Network)

[HRMG.py](file:///c:/Users/ZhuanZ/Desktop/PMTI-HRMG/model/HRMG.py) is the core novel model of this project. Its architecture is as follows:

1. **Input projection layer**: Projects raw node features into an appropriate dimensional space
2. **Multi-scale spectral convolution branches**: Parallel ChebConv layers with different K values extract multi-scale information
3. **Scale attention mechanism**: Automatically learns attention weights for each node across different scales
4. **GCN layer**: Performs further graph convolution on attention-fused features
5. **JK connection**: Optional jumping knowledge connection for fusing multi-layer features
6. **MLP prediction head**: Concatenates embeddings of node pairs for link prediction

## Evaluation Metrics

Model performance is comprehensively evaluated using the following metrics:

- **AUC (Area Under ROC Curve)**
- **AUPR (Area Under Precision-Recall Curve)**
- **Precision**
- **Recall**
- **Accuracy**
- **F1-Score**

## Citation

If you use the code or method of this project in your research, please cite this work appropriately.

## License

This project is for academic research use only.
