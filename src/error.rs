use thiserror::Error;

/// Errors that can occur when working with HMM models.
#[derive(Debug, Error)]
pub enum HmmError {
    #[error("Invalid number of states: {0} (must be > 0)")]
    InvalidStateCount(usize),

    #[error("Invalid feature dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Probability vector must sum to 1.0 (got {sum})")]
    InvalidProbabilitySum { sum: f64 },

    #[error("Transition matrix row {row} must sum to 1.0 (got {sum})")]
    InvalidTransitionRow { row: usize, sum: f64 },

    #[error("Probability must be positive (got {value} at index {index})")]
    InvalidProbability { index: usize, value: f64 },

    #[error("Transition probability must be positive (got {value} at row {row}, col {col})")]
    InvalidTransitionProbability { row: usize, col: usize, value: f64 },

    #[error("Variance must be positive (got {value} at state {state}, feature {feature})")]
    InvalidVariance {
        state: usize,
        feature: usize,
        value: f64,
    },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Model file not found: {0}")]
    ModelFileNotFound(String),

    #[error("Invalid model format: {0}")]
    InvalidModelFormat(String),
}
