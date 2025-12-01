# GRNimmuneClock

**Cell-Type Specific Aging Clocks for Immune Cells**

GRNimmuneClock provides pre-trained aging clocks for immune cell types, built using gene regulatory network (GRN) analysis. Predict biological age from gene expression data with cell-type specific models trained on multiple cohorts.

## Features

- 🔬 **Cell-Type Specific**: Separate models for CD4T and CD8T cells
- 📊 **High Performance**: Trained on multiple cohorts with Spearman corr > 0.8.
- 🧬 **GRN-Based**: Uses gene regulatory network-informed features
- 🔗 **Network Analysis**: Access GRNs for TF-target exploration
- 🎨 **Visualization Tools**: Built-in plotting functions for analysis
- 🚀 **Easy to Use**: Simple Python API
- 🔧 **Training Pipeline**: Tools to train custom aging clocks

## Installation

```bash
pip install grnimmuneclock
```

Or install from source:

```bash
git clone https://github.com/janursa/GRNimmuneClock.git
cd GRNimmuneClock
pip install -e .
```

## Quick Start

```python
from grnimmuneclock import AgingClock, load_example_data
import grnimmuneclock.plotting as gplot

# Load pre-trained clock for CD4T cells
clock = AgingClock(cell_type='CD4T')

# Load example data
adata = load_example_data()

# Predict biological age
adata_predicted = clock.predict(adata)
print(adata_predicted.obs['predicted_age'])

# Visualize predictions
gplot.plot_predicted_vs_actual(adata_predicted, hue='sex')
```
See the tutorial.ipynb for more.

## Supported Cell Types

- `CD4T`: CD4+ T cells 
- `CD8T`: CD8+ T cells 


## Model Information

All models are:
- **Algorithm**: Ridge regression with StandardScaler
- **Features**: Gene expression values (target genes from GRN analysis)
- **Training**: Multiple cohorts (European, Korean, Japanese, Chinese)
- **Age Range**: 20-80 years
- **Species**: Human
- **Tissue**: Peripheral blood


## Citation

If you use GRNimmuneClock in your research, please cite:

```bibtex
@article{nourisa2025grnimmuneclock,
  title={TBD},
  author={Nourisa, Jalil and others},
  journal={TBD},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and issues, please open an issue on [GitHub](https://github.com/janursa/GRNimmuneClock/issues).

