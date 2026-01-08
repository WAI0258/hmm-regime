pub mod error;
mod forward_backward;
pub mod model;

pub use error::HmmError;
pub use model::GaussianHmm;
