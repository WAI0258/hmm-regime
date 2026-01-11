use crate::error::HmmError;
use crate::utils;

/// Forward-only algorithm for HMM inference (log-space).
/// 1. **Forward pass**: Compute α_t(i) = log P(o_1, ..., o_t, state_t = i)
/// 2. **Normalize**: P(state_t | o_1, ..., o_t) = normalize(exp(α_t))
pub fn forward_only(
    observations: &[Vec<f64>],
    initial_probs: &[f64],
    transition_matrix: &[Vec<f64>],
    emission_means: &[Vec<f64>],
    emission_variances: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, HmmError> {
    let n_states = initial_probs.len();
    let t = observations.len();

    if t == 0 {
        return Ok(vec![]);
    }

    let n_features = observations[0].len();
    if n_features == 0 {
        return Err(HmmError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }

    if transition_matrix.len() != n_states {
        return Err(HmmError::DimensionMismatch {
            expected: n_states,
            actual: transition_matrix.len(),
        });
    }
    if emission_means.len() != n_states || emission_variances.len() != n_states {
        return Err(HmmError::DimensionMismatch {
            expected: n_states,
            actual: emission_means.len(),
        });
    }

    if let Some((index, &value)) = initial_probs
        .iter()
        .enumerate()
        .find(|(_, &p)| p < 0.0 || p > 1.0)
    {
        return Err(HmmError::InvalidProbability { index, value });
    }

    for (i, row) in transition_matrix.iter().enumerate() {
        if let Some((col, &value)) = row.iter().enumerate().find(|(_, &p)| p < 0.0 || p > 1.0) {
            return Err(HmmError::InvalidTransitionProbability { row: i, col, value });
        }
    }

    // Forward pass: compute alpha (log-space)
    let mut alpha = vec![vec![0.0; n_states]; t];

    // Initialize: α_0(i) = log(π_i) + log(P(o_0 | state_i))
    for i in 0..n_states {
        let log_emission = utils::multivariate_gaussian_log_pdf_diagonal(
            &observations[0],
            &emission_means[i],
            &emission_variances[i],
            i,
        )?;
        alpha[0][i] = initial_probs[i].ln() + log_emission;
    }

    // Normalize first time step
    let log_sum = utils::log_sum_exp(&alpha[0]);
    for prob in alpha[0].iter_mut() {
        *prob -= log_sum;
    }

    // Forward recursion: α_t(i) = log(Σ_j exp(α_{t-1}(j) + log(A_{ji}))) + log(P(o_t | state_i))
    for t_idx in 1..t {
        for i in 0..n_states {
            let mut log_sum_terms = Vec::with_capacity(n_states);
            #[allow(clippy::needless_range_loop)]
            for j in 0..n_states {
                let log_transition = transition_matrix[j][i].ln();
                log_sum_terms.push(alpha[t_idx - 1][j] + log_transition);
            }
            let log_emission = utils::multivariate_gaussian_log_pdf_diagonal(
                &observations[t_idx],
                &emission_means[i],
                &emission_variances[i],
                i,
            )?;
            alpha[t_idx][i] = utils::log_sum_exp(&log_sum_terms) + log_emission;
        }

        // Normalize to prevent underflow and convert to probabilities
        let log_sum = utils::log_sum_exp(&alpha[t_idx]);
        for prob in alpha[t_idx].iter_mut() {
            *prob -= log_sum;
        }
    }

    // Convert from log-space to probability space: P(state_t | o_1, ..., o_t) = exp(α_t)
    let mut probabilities = vec![vec![0.0; n_states]; t];
    for t_idx in 0..t {
        let max_log = alpha[t_idx]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = alpha[t_idx].iter().map(|&lg| (lg - max_log).exp()).sum();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_states {
            probabilities[t_idx][i] = (alpha[t_idx][i] - max_log).exp() / sum;
        }
    }

    Ok(probabilities)
}
