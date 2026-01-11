use crate::error::HmmError;
use crate::forward::forward_only;
use serde::{Deserialize, Serialize};
use std::path::Path;

const PROB_EPSILON: f64 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GaussianHmm {
    n_states: usize,
    n_features: usize,
    initial_probs: Vec<f64>,
    transition_matrix: Vec<Vec<f64>>,
    emission_means: Vec<Vec<f64>>,
    #[serde(alias = "emission_covs")]
    emission_variances: Vec<Vec<f64>>,
}

impl GaussianHmm {
    pub fn new(
        initial_probs: Vec<f64>,
        transition_matrix: Vec<Vec<f64>>,
        emission_means: Vec<Vec<f64>>,
        emission_variances: Vec<Vec<f64>>,
    ) -> Result<Self, HmmError> {
        let n_states = initial_probs.len();
        if n_states == 0 {
            return Err(HmmError::InvalidStateCount(0));
        }

        // Validate initial probabilities (allow zero probabilities, log-space can handle them)
        for (i, &prob) in initial_probs.iter().enumerate() {
            if prob < 0.0 || prob > 1.0 {
                return Err(HmmError::InvalidProbability {
                    index: i,
                    value: prob,
                });
            }
        }
        let prob_sum: f64 = initial_probs.iter().sum();
        if (prob_sum - 1.0).abs() > PROB_EPSILON {
            return Err(HmmError::InvalidProbabilitySum { sum: prob_sum });
        }

        // Validate transition matrix
        if transition_matrix.len() != n_states {
            return Err(HmmError::DimensionMismatch {
                expected: n_states,
                actual: transition_matrix.len(),
            });
        }

        for (i, row) in transition_matrix.iter().enumerate() {
            if row.len() != n_states {
                return Err(HmmError::DimensionMismatch {
                    expected: n_states,
                    actual: row.len(),
                });
            }
            // Allow zero transition probabilities (log-space can handle them)
            for (j, &prob) in row.iter().enumerate() {
                if prob < 0.0 || prob > 1.0 {
                    return Err(HmmError::InvalidTransitionProbability {
                        row: i,
                        col: j,
                        value: prob,
                    });
                }
            }
            let row_sum: f64 = row.iter().sum();
            if (row_sum - 1.0).abs() > PROB_EPSILON {
                return Err(HmmError::InvalidTransitionRow {
                    row: i,
                    sum: row_sum,
                });
            }
        }

        // Validate emission parameters
        if emission_means.len() != n_states {
            return Err(HmmError::DimensionMismatch {
                expected: n_states,
                actual: emission_means.len(),
            });
        }

        if emission_variances.len() != n_states {
            return Err(HmmError::DimensionMismatch {
                expected: n_states,
                actual: emission_variances.len(),
            });
        }

        let n_features = emission_means[0].len();
        if n_features == 0 {
            return Err(HmmError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }

        for (i, (mean_row, var_row)) in emission_means
            .iter()
            .zip(emission_variances.iter())
            .enumerate()
        {
            if mean_row.len() != n_features {
                return Err(HmmError::DimensionMismatch {
                    expected: n_features,
                    actual: mean_row.len(),
                });
            }
            if var_row.len() != n_features {
                return Err(HmmError::DimensionMismatch {
                    expected: n_features,
                    actual: var_row.len(),
                });
            }

            // Validate variances are positive
            for (j, &var) in var_row.iter().enumerate() {
                if var <= 0.0 {
                    return Err(HmmError::InvalidVariance {
                        state: i,
                        feature: j,
                        value: var,
                    });
                }
            }
        }

        Ok(Self {
            n_states,
            n_features,
            initial_probs,
            transition_matrix,
            emission_means,
            emission_variances,
        })
    }

    /// Load model from JSON file (Python-trained model).
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self, HmmError> {
        let path_ref = path.as_ref();
        let content = std::fs::read_to_string(path_ref).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                HmmError::ModelFileNotFound(path_ref.display().to_string())
            } else {
                HmmError::IoError(e)
            }
        })?;

        let mut model: Self = serde_json::from_str(&content)?;

        // Infer n_features if missing (backward compatibility)
        if model.n_features == 0 && !model.emission_means.is_empty() {
            model.n_features = model.emission_means[0].len();
        }

        // Validate loaded model
        model.validate()?;

        Ok(model)
    }

    /// Save model to JSON file (for debugging).
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> Result<(), HmmError> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Get number of states.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Get number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get emission means for all states.
    ///
    /// Returns a slice of emission mean vectors, one per state.
    /// Each vector has length `n_features`.
    pub fn emission_means(&self) -> &[Vec<f64>] {
        &self.emission_means
    }

    /// Infer regime probabilities for a single observation.
    pub fn infer_single(&self, observation: &[f64]) -> Result<Vec<f64>, HmmError> {
        if observation.len() != self.n_features {
            return Err(HmmError::DimensionMismatch {
                expected: self.n_features,
                actual: observation.len(),
            });
        }

        let observations = vec![observation.to_vec()];
        let result = forward_only(
            &observations,
            &self.initial_probs,
            &self.transition_matrix,
            &self.emission_means,
            &self.emission_variances,
        )?;

        Ok(result[0].clone())
    }

    /// Infer regime probabilities for a sequence of observations.
    pub fn infer_sequence(&self, observations: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, HmmError> {
        for obs in observations {
            if obs.len() != self.n_features {
                return Err(HmmError::DimensionMismatch {
                    expected: self.n_features,
                    actual: obs.len(),
                });
            }
        }

        forward_only(
            observations,
            &self.initial_probs,
            &self.transition_matrix,
            &self.emission_means,
            &self.emission_variances,
        )
    }

    /// Validate model parameters.
    fn validate(&self) -> Result<(), HmmError> {
        // Quick validation: check basic constraints without cloning
        if self.n_states == 0 {
            return Err(HmmError::InvalidStateCount(0));
        }
        if self.n_features == 0 {
            return Err(HmmError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }

        // Check probability sums (quick check)
        let prob_sum: f64 = self.initial_probs.iter().sum();
        if (prob_sum - 1.0).abs() > PROB_EPSILON {
            return Err(HmmError::InvalidProbabilitySum { sum: prob_sum });
        }
        Ok(())
    }
}
