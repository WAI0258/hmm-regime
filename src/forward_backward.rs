use crate::error::HmmError;

/// Minimum variance to prevent numerical instability.
const MIN_VARIANCE: f64 = 1e-10;

/// standard trick: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Compute multivariate Gaussian log PDF with diagonal covariance.
/// log_pdf = Σ_i log(N(x_i | μ_i, σ_i²))
///
/// Where N(x | μ, σ²) is the univariate Gaussian PDF:
/// log(N(x | μ, σ²)) = -0.5 * ln(2π) - 0.5 * ln(σ²) - 0.5 * (x - μ)² / σ²
/// Log probability density (log-space)
pub fn multivariate_gaussian_log_pdf_diagonal(
    observation: &[f64],
    mean: &[f64],
    variances: &[f64],
    state: usize,
) -> Result<f64, HmmError> {
    if observation.len() != mean.len() || observation.len() != variances.len() {
        return Err(HmmError::DimensionMismatch {
            expected: mean.len(),
            actual: observation.len(),
        });
    }

    let mut log_pdf = 0.0;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();

    for (i, ((&x, &mu), &sigma_sq)) in observation
        .iter()
        .zip(mean.iter())
        .zip(variances.iter())
        .enumerate()
    {
        if sigma_sq <= 0.0 {
            return Err(HmmError::InvalidVariance {
                state,
                feature: i,
                value: sigma_sq,
            });
        }

        // Clamp very small variances to prevent numerical instability
        let sigma_sq_clamped = sigma_sq.max(MIN_VARIANCE);
        let diff = x - mu;
        log_pdf -= 0.5 * (log_2pi + sigma_sq_clamped.ln() + (diff * diff) / sigma_sq_clamped);
    }

    Ok(log_pdf)
}

/// Forward-Backward algorithm for HMM inference (log-space).
///
/// Computes posterior probabilities P(state_t | observations) for each time step.
///
/// # Algorithm
///
/// 1. **Forward pass**: Compute α_t(i) = log P(o_1, ..., o_t, state_t = i)
/// 2. **Backward pass**: Compute β_t(i) = log P(o_{t+1}, ..., o_T | state_t = i)
/// 3. **Gamma**: Compute γ_t(i) = P(state_t = i | observations) using α and β
pub fn forward_backward(
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

    // Validate dimensions
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
    // Allow zero probabilities (log-space can handle log(0) = -inf)
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
        let log_emission = multivariate_gaussian_log_pdf_diagonal(
            &observations[0],
            &emission_means[i],
            &emission_variances[i],
            i,
        )?;
        alpha[0][i] = initial_probs[i].ln() + log_emission;
    }

    // Helper function to normalize log probabilities in-place
    let normalize_log_probs = |probs: &mut [f64]| {
        let log_sum = log_sum_exp(probs);
        for prob in probs.iter_mut() {
            *prob -= log_sum;
        }
    };

    // Normalize first time step
    normalize_log_probs(&mut alpha[0]);

    // Forward recursion: α_t(i) = log(Σ_j exp(α_{t-1}(j) + log(A_{ji}) + log(P(o_t | state_i))))
    for t_idx in 1..t {
        for i in 0..n_states {
            let mut log_sum_terms = Vec::with_capacity(n_states);
            #[allow(clippy::needless_range_loop)]
            for j in 0..n_states {
                let log_transition = transition_matrix[j][i].ln();
                log_sum_terms.push(alpha[t_idx - 1][j] + log_transition);
            }
            let log_emission = multivariate_gaussian_log_pdf_diagonal(
                &observations[t_idx],
                &emission_means[i],
                &emission_variances[i],
                i,
            )?;
            alpha[t_idx][i] = log_sum_exp(&log_sum_terms) + log_emission;
        }

        // Normalize to prevent underflow
        normalize_log_probs(&mut alpha[t_idx]);
    }

    // Backward pass: compute beta (log-space)
    let mut beta = vec![vec![0.0; n_states]; t];

    // Initialize: β_{T-1}(i) = 0 (log(1) = 0)
    for i in 0..n_states {
        beta[t - 1][i] = 0.0;
    }

    // Backward recursion: β_t(i) = log(Σ_j exp(log(A_{ij}) + log(P(o_{t+1} | state_j)) + β_{t+1}(j)))
    for t_idx in (0..t - 1).rev() {
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_states {
            let mut log_sum_terms = Vec::with_capacity(n_states);
            for j in 0..n_states {
                let log_transition = transition_matrix[i][j].ln();
                let log_emission = multivariate_gaussian_log_pdf_diagonal(
                    &observations[t_idx + 1],
                    &emission_means[j],
                    &emission_variances[j],
                    j,
                )?;
                log_sum_terms.push(log_transition + log_emission + beta[t_idx + 1][j]);
            }
            beta[t_idx][i] = log_sum_exp(&log_sum_terms);
        }

        // Normalize to prevent underflow
        normalize_log_probs(&mut beta[t_idx]);
    }

    // Compute gamma: γ_t(i) = P(state_t = i | observations) = exp(α_t(i) + β_t(i) - log_likelihood)
    // The log-likelihood is the normalization constant: log P(observations) = log_sum_exp(α_T)
    let log_likelihood = log_sum_exp(&alpha[t - 1]);
    let mut gamma = vec![vec![0.0; n_states]; t];

    for t_idx in 0..t {
        let mut log_gamma = Vec::with_capacity(n_states);
        for i in 0..n_states {
            log_gamma.push(alpha[t_idx][i] + beta[t_idx][i] - log_likelihood);
        }

        // Convert from log-space to probability space and normalize
        let max_log = log_gamma.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = log_gamma.iter().map(|&lg| (lg - max_log).exp()).sum();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_states {
            gamma[t_idx][i] = (log_gamma[i] - max_log).exp() / sum;
        }
    }

    Ok(gamma)
}
