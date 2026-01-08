# hmm-regime

A Rust library for Hidden Markov Model (HMM) regime classification with Gaussian emissions.

## Features

- **Gaussian HMM**: Implements Hidden Markov Models with Gaussian emission distributions
- **Forward-Backward Algorithm**: Efficient inference using log-space computations to prevent numerical underflow
- **Regime Classification**: Infer regime probabilities for single observations or sequences
- **Model Persistence**: Load and save models in JSON format (compatible with Python-trained models)
- **Robust Validation**: Comprehensive parameter validation with detailed error messages

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
hmm-regime = "0.1.0"
```

### Basic Example

```rust
use hmm-regime::GaussianHmm;

// Create a model with 2 states and 1 feature
let initial_probs = vec![0.5, 0.5];
let transition_matrix = vec![
    vec![0.9, 0.1],
    vec![0.1, 0.9],
];
let emission_means = vec![
    vec![0.0],  // State 0 mean
    vec![5.0],  // State 1 mean
];
let emission_variances = vec![
    vec![1.0],  // State 0 variance
    vec![1.0],  // State 1 variance
];

let model = GaussianHmm::new(
    initial_probs,
    transition_matrix,
    emission_means,
    emission_variances,
)?;

// Infer regime for a single observation
let observation = vec![2.5];
let probabilities = model.infer_single(&observation)?;
println!("Regime probabilities: {:?}", probabilities);

// Infer regime for a sequence
let observations = vec![
    vec![0.5],
    vec![1.0],
    vec![4.5],
    vec![5.5],
];
let sequence_probs = model.infer_sequence(&observations)?;
```

### Loading a Pre-trained Model

```rust
// Load model from JSON file (e.g., trained in Python)
let model = GaussianHmm::load_from_json("model.json")?;

// Use the model for inference
let probabilities = model.infer_single(&observation)?;
```

## API

### `GaussianHmm`

The main model struct that encapsulates HMM parameters and provides inference methods.

#### Methods

- `new()`: Create a new HMM model with validation
- `load_from_json()`: Load a model from a JSON file
- `save_to_json()`: Save a model to a JSON file
- `infer_single()`: Infer regime probabilities for a single observation
- `infer_sequence()`: Infer regime probabilities for a sequence of observations
- `n_states()`: Get the number of states
- `n_features()`: Get the number of features
- `emission_means()`: Get emission means for all states

## Error Handling

The library uses `HmmError` for comprehensive error reporting:

- Invalid state counts or dimensions
- Invalid probability distributions
- Invalid transition matrices
- Invalid variance values
- I/O errors when loading/saving models
- JSON parsing errors

## Algorithm

The library implements the Forward-Backward algorithm in log-space to compute posterior probabilities:

1. **Forward pass**: Computes α_t(i) = log P(o_1, ..., o_t, state_t = i)
2. **Backward pass**: Computes β*t(i) = log P(o*{t+1}, ..., o_T | state_t = i)
3. **Gamma computation**: Computes γ_t(i) = P(state_t = i | observations)

All computations use log-space arithmetic to prevent numerical underflow.

## License

MIT OR Apache-2.0
