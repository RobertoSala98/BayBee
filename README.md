# ABeeC: Artificial Bee Colony Hybrid Suite
<div align="center">
  <picture>
    <img alt="ABeeC Logo" src="https://github.com/litovn/ABeeC/blob/main/abeec.png?raw=true" width="340">
  </picture>
</div>

> [!IMPORTANT]
> If you use ABeeC in your research, please cite:

```bibtex
@mastersthesis{litovchenko2025,
  author = {Nikita Litovchenko},
  title = {Hybrid Asynchronous Artificial Bee Colony Algorithm for Constrained Optimization in HPC},
  school = {Politecnico di Milano},
  year = {2025},
  note = {Available at \url{https://github.com/litovn/ABeeC}}
}
```
```bibtex
@inproceedings{nikita2025abc,
  title={An Artificial Bee Colony algorithm with Machine Learning for Constrained Optimization in HPC},
  author={Sala, Roberto and Litovchenko, Nikita and Gadioli, Davide and Palermo, Gianluca and Ardagna, Danilo},
  booktitle={MASCOTS},
  pages={1--9},
  year={2025},
  organization={IEEE}
}
```

## Overview

**ABeeC** (Artificial Bee Colony - Hybrid Suite) is an optimization framework that extends the classical Artificial Bee Colony algorithm, for solving constrained, discrete optimization problems in high-performance computing (HPC) environments. This project is the result of a Master's thesis in Computer Science and Engineering at Politecnico di Milano.

The framework addresses critical limitations of traditional optimization approaches in HPC by introducing **asynchronous event-driven execution**, **machine learning-guided search**, and **hybrid Bayesian optimization** integration. Unlike standard ABC implementations designed for continuous unconstrained problems, ABeeC is engineered for real-world HPC scenarios involving expensive black-box functions, discrete configuration spaces, and strict computational constraints.

### Key Features

- ✅ **Three Advanced Variants**: ABC-MLCO, ABC-AMLO, and hybrid BO integration
- ✅ **Asynchronous Execution**: Event-driven architecture eliminating synchronization barriers
- ✅ **Black-Box Constraint Handling**: Intelligent constraint management without analytical models
- ✅ **Memory-Based Optimization**: Local and global memory mechanisms with Bloom Filter caching
- ✅ **ML-Guided Search**: Surrogate models (Ridge Regression, Random Forest, Neural Networks) for directed exploration
- ✅ **Lévy Flight Exploration**: Adaptive long-range jumps to escape local optima
- ✅ **Bayesian Optimization Integration**: Hybrid approach combining global and local search

## Algorithm Variants

### ABC-MLCO: Machine Learning for Constrained Optimization

**Purpose**: Enhance the original ABC algorithm for black-box constrained optimization in discrete domains.

**Innovations**:
- Feasibility-aware fitness function steering search away from infeasible regions
- ML surrogate models trained on individual bee histories for exploitation
- Lévy flight distribution for enhanced exploration with adaptive probability
- Memory-based duplicate detection to reduce redundant evaluations from ~60% to ~10%

### ABC-AMLO: Asynchronous Machine Learning and Optimization

**Purpose**: Maximize computational resource utilization through event-driven execution.

**Innovations**:
- Eliminates three synchronization barriers inherent in synchronous ABC
- Event-driven manager using priority queues for continuous task scheduling
- Dual memory strategy: local memory for focused search + global memory for collaborative knowledge
- Bloom Filter-based cache for efficient duplicate detection in highly parallel settings

### Hybrid BO Integration

**Purpose**: Combine ABC's global exploration with Bayesian Optimization's local refinement.

**Strategy**: Probabilistically triggered BO activation increasing over optimization stages

**Synergy**: Prevents BO premature convergence while accelerating ABC's late-stage refinement.

## Real-World Validation:

The framework was validated on a molecular docking application from the EXSCALATE virtual screening platform, addressing a production HPC auto-tuning scenario with:

- **Problem Scale**: 8-dimensional discrete configuration space
- **Constraint Types**: Execution time limits, minimum accuracy thresholds
- **Evaluation Cost**: ~9 minutes per configuration evaluation

## Research Contributions

This work addresses several open problems in swarm intelligence and HPC optimization:

1. **First Asynchronous ABC Implementation for HPC**: ABC-AMLO replaces synchronous iteration model with event-driven execution, achieving significant speedups
2. **Memory-Based Discrete Optimization**: Adapts ABC for discrete domains with memory-guided duplicate avoidance
3. **Constraint-Aware Surrogate Models**: Integrates ML prediction of black-box constraints to steer search toward feasible regions
4. **Hybrid Global-Local Search**: Seamlessly combines ABC's robust global exploration with Bayesian Optimization's efficient local refinement
5. **Production HPC Validation**: Real-world case study on molecular docking application demonstrates practical impact

## Installation

### Dependencies

Core packages:
- `numpy` - Efficient numerical operations
- `scikit-learn` - ML models and utilities
- `scipy` - Scientific computing functions
- `pandas` - Data manipulation and analysis
- `xgboost` - Gradient boosting for surrogate models
- `matplotlib` - Visualization and result plotting

See `requirements.txt` for complete specifications.

### Setup
```bash
# Clone the repository
git clone https://github.com/litovn/ABeeC.git
cd ABeeC

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ABeeC/
├── README.md
├── requirements.txt
├── config.json                 # Configuration template
├── runabeec.py                 # Main entry point
│
├── src/
│   ├── coreabc.py              # Core ABC implementation (Beehive, Bee classes)
│   ├── simulate.py             # Evaluation and simulation module
│   ├── bomanager.py            # Bayesian Optimization manager
│   │
│   ├── dMALIBOO/               # Bayesian Optimization framework
│   │   ├── d_maliboo.py
│   │   └── search_space.py
│   │
│   ├── strategy/               # Algorithm components
│   │   ├── levy.py             # Lévy flight implementation
│   │   ├── memory.py           # Memory management (local/global)
│   │   ├── surrogate.py        # ML surrogate model training
│   │   └── bo.py               # BO integration logic
│   │
│   └── utils/                  # Utility functions
│       ├── config.py           # Configuration parsing
│       ├── dataset.py          # Data loading and preprocessing
│       ├── cache.py            # Bloom Filter cache implementation
│       ├── logging.py          # Activity logging for analysis
│       └── plotting.py         # Result visualization
│
└── tests/                      # Unit and integration tests
    ├── test_core.py
    └── test_algorithms.py
```

## Architecture and Design

### Core Classes

**Beehive**: Population manager maintaining all bees and orchestrating optimization phases
- Manages bee population lifecycle
- Coordinates role transitions (employed → onlooker → scout)
- Tracks global best solution and convergence metrics

**Bee**: Individual agent representing a candidate solution
- Maintains local memory of evaluated configurations
- Trains surrogate models on exploration history
- Computes fitness based on objective and constraint functions
- Tracks trial counter for stagnation detection

**Event-Driven Manager** (ABC-AMLO only)
- Priority queue of bee tasks ordered by completion time
- Processes results and schedules next operations
- Manages transition between optimization phases
- Coordinates cache and duplicate detection

**Bloom Filter Cache** (ABC-AMLO only)
- Probabilistic data structure for efficient duplicate detection
- Configurable false-positive rate (parameter ε)
- Reduces memory overhead compared to storing all evaluations

### Algorithm Flow

```
Initialization
    ↓
[ABC-AMLO Event-Driven Loop]
    ├─ Event Manager: Select next bee to process
    ├─ Phase Transition: Based on previous role (Scout→Employed→Onlooker→Scout)
    │
    ├─ Employed Phase
    │   ├─ Train surrogate model on bee's memory
    │   ├─ Generate candidate via ML-guided neighborhood search
    │   ├─ Evaluate objective and constraint functions
    │   └─ Update best position and trial counter
    │
    ├─ Onlooker Phase
    │   ├─ Compute roulette wheel selection probabilities
    │   ├─ Probabilistically choose: Lévy flight OR roulette selection
    │   ├─ Generate candidate around selected bee
    │   └─ Update position and trial counter
    │
    ├─ Scout Phase
    │   ├─ Check if trial limit exceeded
    │   ├─ Train constraint prediction model
    │   ├─ Sample new position from feasible region
    │   └─ Reset trial counter
    │
    ├─ Bayesian Optimization (probabilistic trigger)
    │   ├─ Check activation probability based on stage
    │   ├─ Train Gaussian Process surrogate
    │   ├─ Compute acquisition function
    │   └─ Perform local refinement steps
    │
    └─ Cache Management
        ├─ Check Bloom Filter for duplicates
        ├─ Avoid redundant evaluations
        └─ Update global memory if using global strategy

Termination Condition Met
    ↓
Return Best Feasible Solution
```

## Key Parameters and Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 50 | Number of bees (population size) |
| `T` | 50-100 | Maximum iterations |
| `limit` | 10 | Trial threshold before scout phase |
| `s_levy` | 0.1 | Lévy flight step-size scaling |
| `levy_β` | 1.5 | Heavy-tailed exponent (1.0-2.0) |
| `H_min` | 10 | Minimum history for surrogate training |
| `H_max` | 200 | Maximum history size per bee |
| `memory_type` | "global" | "local" for focused search, "global" for collaborative |
| `surrogate_model` | "ridge" | "ridge", "random_forest", or "neural_network" |
| `use_bayesian_optimization` | true | Enable hybrid BO integration |

### Tuning Guidelines

- **Exploration vs. Exploitation**: Increase `s_levy` or decrease `limit` to favor exploration
- **Convergence Speed**: Use "global" memory for faster initial convergence; switch to "local" for fine-tuning
- **Large Search Spaces**: Increase `N` (population size) and enable BO for late-stage refinement
- **Expensive Evaluations**: Use "ridge" regression (faster training) on limited data; switch to "random_forest" as data accumulates
- **Tight Constraints**: Lower `levy_β` to favor smaller steps and feasibility preservation

## Performance Comparison

### Against Baselines

| Algorithm | LiGen MAPR | Feasibility | Convergence Speed |
|-----------|-----------|-------------|------------------|
| OpenTuner | baseline | 28% | baseline |
| ABC (original) | -2793% | 20% | baseline |
| ABC-MLCO | -2851% | 50% | -50% vs ABC |
| ABC-AMLO | -2882% | 54% | 3.6× faster |

*Lower MAPR is better (negative values show improvement)*

## Future Work and Extensions

Potential areas for extension:

1. **Multi-Objective Optimization**: Extend to Pareto-front discovery for trade-off exploration
2. **Distributed Computing**: Scale to multi-node clusters with MPI integration
3. **Adaptive Hyperparameters**: Self-tuning of population size, Lévy parameters
4. **GPU Acceleration**: Surrogate model training and acquisition function optimization on CUDA
5. **Transfer Learning**: Warm-start from similar optimization problems
6. **Real-Time Control**: Applications to dynamic systems and online optimization

## Limitations and Known Issues

- **Current Limitations**:
  - Surrogate models limited to ~500 sample points per bee for memory efficiency
  - Bloom Filter false-positive rate increases with cache size; tunable via ε parameter
  - Bayesian Optimization integration assumes relatively small numbers of BO steps (~1-3)


## License

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.

