# IBL Psychedelics Project Reorganization Plan

**Date Created:** 2025-12-03

---

## Table of Contents
1. [Current Issues](#current-issues)
2. [Proposed Directory Structure](#proposed-directory-structure)
3. [Reorganization Strategy](#reorganization-strategy)
4. [Code Improvement Ideas](#code-improvement-ideas)
5. [Novel Analysis Suggestions](#novel-analysis-suggestions)

---

## Current Issues

### Critical Problems
1. **Duplicate notebooks** scattered across root, `davide/`, `martin/`, and `archive/`
2. **Personal directories** (davide/, martin/, olivier/) mixed with core codebase
3. **Temporary files** in root (`LSD_trajectories.csv`, `rec_temp.csv`)
4. **No dependency management** (no requirements.txt or environment.yml)
5. **Timestamped result files** accumulating in `data/` and `results/`
6. **Large monolithic scripts** (single_unit.py: 914 lines, population_dimensionality.py: 663 lines)
7. **.ipynb_checkpoints** visible despite gitignore
8. **No tests** directory or testing infrastructure
9. **Text todo file** instead of issues/project management
10. **Unclear separation** between exploratory and production code

### Impact
- Hard to onboard new collaborators
- Difficult to find canonical versions of analyses
- Risk of data/code loss
- No reproducibility guarantees
- Slows down development

---

## Proposed Directory Structure

```
ibl_psychedelics/
├── README.md                      # Enhanced documentation
├── setup.py                       # Python package setup
├── requirements.txt               # NEW: Dependency pinning
├── environment.yml                # NEW: Conda environment (optional)
├── pyproject.toml                 # NEW: Modern Python packaging (optional)
│
├── psyfun/                        # Core analysis package
│   ├── __init__.py               # Package initialization
│   ├── io.py                     # Data loading from IBL
│   ├── spikes.py                 # Single-unit metrics
│   ├── population.py             # Population analyses
│   ├── atlas.py                  # Brain region mapping
│   ├── plots.py                  # Plotting utilities
│   ├── util.py                   # General utilities
│   ├── config.py                 # Configuration & paths
│   └── stats.py                  # NEW: Statistical tests
│
├── scripts/                       # NEW: Production analysis scripts
│   ├── preprocessing/
│   │   ├── fetch_data.py         # MOVE from root
│   │   └── dataset_overview.py   # MOVE from root
│   ├── single_neuron/
│   │   └── run_single_unit_analysis.py
│   ├── population/
│   │   └── run_population_dimensionality.py
│   └── figures/
│       ├── plot_single_unit_results.py
│       └── plot_population_results.py
│
├── notebooks/                     # NEW: Organized notebooks
│   ├── 00_data_acquisition/
│   │   └── download_data.ipynb   # Canonical version
│   ├── 01_exploratory/
│   │   ├── single_unit_exploration.ipynb
│   │   ├── PCA_exploration.ipynb
│   │   └── brain_states_exploration.ipynb
│   ├── 02_figures/               # Publication-ready
│   │   ├── figure1_dataset_overview.ipynb
│   │   ├── figure2_single_neuron.ipynb
│   │   └── figure3_population.ipynb
│   └── README.md                 # Notebook index
│
├── analyses/                      # NEW: Personal/experimental
│   ├── davide/                   # MOVE personal work here
│   ├── martin/                   # MOVE personal work here
│   └── olivier/                  # MOVE personal work here
│
├── tests/                         # NEW: Unit tests
│   ├── __init__.py
│   ├── test_io.py
│   ├── test_spikes.py
│   ├── test_population.py
│   └── test_util.py
│
├── data/                          # Data files (.gitignore)
│   ├── raw/                      # NEW: Raw downloads
│   ├── processed/                # NEW: Preprocessed
│   │   ├── spikes.h5
│   │   └── units.pqt
│   └── README.md                 # Data documentation
│
├── metadata/                      # Session/probe metadata
│   ├── sessions.pqt
│   ├── insertions.pqt
│   └── trajectories.csv
│
├── results/                       # Analysis outputs
│   ├── single_neuron/            # NEW: Organized by type
│   ├── population/
│   └── statistics/
│
├── figures/                       # Publication figures
│   ├── draft/                    # NEW: Work in progress
│   └── final/                    # NEW: Publication-ready
│
├── archive/                       # Historical code
│   └── README.md                 # Documentation
│
├── docs/                          # NEW: Documentation
│   ├── installation.md
│   ├── data_pipeline.md
│   ├── analysis_guide.md
│   ├── reorganization_plan.md    # This file
│   └── api_reference.md
│
└── .github/                       # NEW: GitHub config
    ├── workflows/
    │   └── tests.yml             # CI/CD
    └── ISSUE_TEMPLATE/
```

---

## Reorganization Strategy

### Infrastructure Setup
- Create `requirements.txt` to pin all dependencies
- Update `.gitignore` to properly exclude checkpoint files, data, and temporary files
- Create all necessary directory structure
- Set up pytest configuration
- Add GitHub Actions for continuous integration

### File Reorganization
- Move preprocessing scripts from root to `scripts/preprocessing/`
- Consolidate duplicate notebooks, keeping one canonical version in `notebooks/`
- Move personal/experimental work to `analyses/<name>/`
- Clean up temporary CSV files from root directory
- Organize timestamped results into subdirectories by analysis type
- Create "latest" symlinks for most recent results

### Code Modularization
- Extract statistical functions from large scripts into `psyfun/stats.py`
- Split monolithic analysis scripts into:
  - Data loading/processing
  - Metric computation
  - Statistical testing
  - Figure generation (separate files)
- Add CLI interfaces to scripts for better usability
- Implement proper logging instead of print statements
- Create versioning system for results files

### Testing Infrastructure
- Create unit tests for all core functions in `psyfun/`
- Test spike metrics (CV, Fano factor, LZ complexity)
- Test data loading functions
- Test statistical functions
- Set up pytest configuration
- Add continuous integration with GitHub Actions

### Documentation
- Enhance main README with installation, usage, and project overview
- Document data pipeline flow
- Create analysis guide explaining each script
- Add API reference for `psyfun` package
- Document project structure and organization

---

## Code Improvement Ideas

### A. Modularize Large Scripts
**Current:** `single_unit.py` is 914 lines in a single file

**Proposed:**
- Extract statistical tests to `psyfun/stats.py` (reusable across analyses)
- Separate plotting code into `scripts/figures/plot_single_unit_results.py`
- Keep main analysis logic streamlined in `scripts/single_neuron/run_single_unit_analysis.py`
- Create helper functions for common operations

### B. Better Configuration Management
**Current:** Hardcoded paths and parameters scattered throughout code

**Proposed:**
- Use dataclass-based configuration system
- Allow environment variable overrides for paths
- Centralize analysis parameters (n_permutations, bin_sizes, etc.)
- Make configuration easily modifiable without editing code

### C. Result Version Management
**Current:** Timestamped files accumulate with no clear "latest" version

**Proposed:**
- Automatic timestamping when saving results
- Create "latest" symlinks pointing to most recent results
- Helper functions to save/load versioned results
- Option to archive old results

### D. Enhanced Package Interface
**Current:** Minimal `__init__.py`, unclear what to import

**Proposed:**
- Proper package initialization with version info
- Clear `__all__` definition for imports
- Expose commonly used functions at package level
- Add package-level documentation

### E. Command-Line Interfaces
**Current:** Scripts use hardcoded parameters

**Proposed:**
- Add CLI arguments for all scripts using `click` or `argparse`
- Allow specifying epochs, metrics, output directories
- Add verbose/quiet modes
- Make scripts more flexible and reusable

### F. Logging System
**Current:** Scattered print statements

**Proposed:**
- Structured logging with different levels (INFO, DEBUG, WARNING)
- Log to both console and file
- Include timestamps and module names
- Better debugging and monitoring

### G. Data Pipeline Clarity
**Current:** Unclear data flow and caching

**Proposed:**
- Clear separation of raw vs processed data
- Explicit caching strategy documented
- Helper functions to check data availability
- Pipeline status checking

---

## Novel Analysis Suggestions

Based on the current codebase capabilities and psychedelics neuroscience literature:

### 1. Neural State Space Dynamics & Trajectories
**Concept:** Analyze temporal trajectories in PC space rather than just static variance explained

**Analyses:**
- Compute path length, curvature, and velocity of neural trajectories
- Measure "trajectory tangling" (how much paths cross in state space)
- Calculate dynamical similarity between pre/post LSD using Procrustes analysis
- Look for fixed points and flow fields in low-dimensional space
- Compare trajectory properties during spontaneous vs stimulus-evoked activity

**Motivation:** Goes beyond static dimensionality to understand temporal dynamics

### 2. Neural Repertoire & Metastability
**Concept:** Quantify the "entropic brain hypothesis" more directly through state space discretization

**Analyses:**
- Cluster neural states using k-means on binned PC projections
- Measure repertoire size (number of distinct states visited)
- Calculate transition probabilities between states
- Measure dwell times in each state (metastability)
- Compute entropy of state occupation distribution
- Analyze switching rate between states pre/post LSD
- Build Markov models of state transitions

**Motivation:** Directly tests whether LSD increases neural state repertoire and flexibility

### 3. Cross-Frequency & Cross-Scale Coupling
**Concept:** Use LFP data to understand how LSD affects multi-scale dynamics

**Analyses:**
- Phase-amplitude coupling: high-frequency spiking coupled to low-frequency LFP
- Spike-phase locking strength and preferred phases
- Cross-frequency directionality using phase-slope index
- Focus on mPFC theta-gamma coupling changes
- Compare coupling strength before/after LSD

**Motivation:** Psychedelics may affect temporal coordination across frequency bands

### 4. Information-Theoretic Measures
**Concept:** Beyond LZ complexity, use information theory to quantify predictability and integration

**Analyses:**
- **Transfer entropy** between brain regions (directed information flow)
- **Integrated Information (Φ)** as proxy for consciousness theories
- **Predictive information**: How much past predicts future activity (may decrease under LSD)
- Time-lagged mutual information to assess information timescales
- Entropy rate of spike trains

**Motivation:** Information theory provides rigorous framework for "entropy" in entropic brain

### 5. Criticality & Scale-Free Dynamics
**Concept:** Test if LSD pushes system toward/away from criticality

**Analyses:**
- **Neuronal avalanche analysis**: Do avalanche size distributions follow power laws?
- Detrended fluctuation analysis (DFA) for temporal scaling
- **Largest Lyapunov exponent**: Quantify chaos/sensitivity to initial conditions
- Hurst exponents for each neuron
- Long-range temporal correlations

**Motivation:** Critical dynamics linked to optimal information processing and consciousness

### 6. Functional Connectivity & Network Topology
**Concept:** Analyze network organization of population activity

**Analyses:**
- Build time-resolved functional connectivity matrices (correlations between neurons)
- Graph theory metrics:
  - Modularity (segregation vs integration)
  - Small-worldness
  - Rich club organization
  - Node centrality changes
- **Dynamic community detection**: Does network organization fluctuate more under LSD?
- Compare connectivity during different behavioral/stimulus epochs

**Motivation:** LSD may alter balance between segregation and integration

### 7. Decoding Stability & Generalization
**Concept:** Test representation stability across conditions

**Analyses:**
- Train decoders on spontaneous00, test on spontaneous01 (representation stability)
- Cross-condition decoding: Train on control, test on LSD (and vice versa)
- Measure decoder confidence calibration
- **Temporal generalization matrices** (train on time t, test on all other times)
- Decoder performance as function of time post-LSD

**Motivation:** Unstable representations would suggest increased flexibility/decreased predictability

### 8. Dimensionality Analysis Extensions
**Concept:** Go beyond counting PCs to explain 80% variance

**Analyses:**
- **Participation ratio** as alternative dimensionality measure
- **Local dimensionality**: Compute in sliding windows to see temporal evolution
- **Manifold capacity** and **abstractness** metrics
- Measure **alignment** between spontaneous and stimulus-evoked manifolds
- Dimensionality of stimulus-specific vs stimulus-invariant subspaces

**Motivation:** Different dimensionality metrics capture different aspects of population coding

### 9. Single-Trial Variability
**Concept:** Analyze response reliability to repeated stimuli

**Analyses:**
- Trial-to-trial variability in replay responses
- Signal vs noise correlations and how they change
- Reliability of individual neurons to repeated stimuli
- Pattern completion analysis
- Fano factor as function of trial number

**Motivation:** LSD may increase variability, reducing stimulus reliability

### 10. Cross-Area Communication Subspaces
**Concept:** Understand how different brain regions communicate

**Analyses:**
- **Communication subspace analysis** between regions (Semedo et al. 2019)
- Canonical correlation analysis (CCA) between regions
- **Information bottleneck**: How much information is shared vs private to each region?
- Granger causality for directional influence
- Dimensionality of communication subspace
- How subspace alignment changes with LSD

**Motivation:** You have multi-area recordings; can test if LSD affects inter-regional communication

### 11. Temporal Correlation Structure
**Concept:** Analyze timescales of neural dynamics

**Analyses:**
- Autocorrelation timescales for each neuron
- Cross-correlation lags between neuron pairs
- Timescale diversity across population
- Changes in temporal receptive fields
- Memory timescales from autoregressive models

**Motivation:** LSD may affect temporal integration windows

### 12. Dimensionality-Behavior Relationships
**Concept:** Link neural dimensionality to behavioral changes

**Analyses:**
- Correlate population dimensionality with behavioral variability
- Does increased neural dimensionality correspond to increased movement diversity?
- Relationship between neural trajectory length and behavioral trajectory length
- Shared dimensionality between neural and behavioral state spaces

**Motivation:** Connects neural and behavioral levels of analysis

---

## Most Tractable Next Steps

Based on existing codebase and analysis pipeline:

### Highly Tractable (Build directly on existing code)
1. **State space trajectories** - Already have PCA, just need to analyze temporal evolution
2. **Neural repertoire** - Straightforward clustering on PC projections
3. **Functional connectivity graphs** - Use existing spike count data

### Moderately Tractable (Require new functions but standard methods)
4. **Dimensionality extensions** - Add participation ratio, local dimensionality
5. **Single-trial variability** - Have repeated stimuli in replay protocol
6. **Decoding stability** - Already thinking about decoders

### More Involved (Require substantial new infrastructure)
7. **Information-theoretic measures** - Need new functions but well-defined
8. **Cross-frequency coupling** - Need to process LFP data (mentioned but not used yet)
9. **Criticality analysis** - Requires avalanche detection infrastructure
10. **Cross-area subspaces** - Need CCA and subspace analysis framework

---

## Priority Quick Wins

These provide immediate benefits with minimal effort:

### File Organization
- Create `requirements.txt` for dependencies
- Update `.gitignore` to fix checkpoint issues
- Delete all `.ipynb_checkpoints/` directories
- Move temporary CSV files out of root
- Create basic new directory structure
- Move personal directories to `analyses/`
- Move scripts to `scripts/` subdirectories

### Code Quality
- Add docstrings to key functions
- Create `psyfun/stats.py` and move statistical functions
- Add basic logging
- Create README sections

### Infrastructure
- Set up pytest
- Create a few basic unit tests
- Add GitHub Actions CI

---

## Key Questions

Before implementing reorganization:

1. **Notebooks:** Which version is "canonical" for each numbered notebook?
2. **Personal directories:** Should `analyses/` be tracked in git or gitignored?
3. **Results:** Should `results/` be git-tracked or gitignored?
4. **Dependencies:** Preference for conda vs pip?
5. **Testing:** Priority level for test coverage?
6. **Novel analyses:** Which should be prioritized first?
7. **Backward compatibility:** Do existing scripts need to keep working during transition?

---

**Last Updated:** 2025-12-03
