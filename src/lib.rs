pub mod error;
mod forward;
mod forward_backward;
mod utils;
pub mod model;

pub use error::HmmError;
pub use model::GaussianHmm;
