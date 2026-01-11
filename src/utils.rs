use crate::error::HmmError;

/// Minimum variance to prevent numerical instability.
pub const MIN_VARIANCE: f64 = 1e-10;

/// Log-space sum-exp trick: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
pub fn log_sum_exp(values: &[f64]) -> f64 {
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

        let sigma_sq_clamped = sigma_sq.max(MIN_VARIANCE);
        let diff = x - mu;
        log_pdf -= 0.5 * (log_2pi + sigma_sq_clamped.ln() + (diff * diff) / sigma_sq_clamped);
    }

    Ok(log_pdf)
}
