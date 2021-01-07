#[derive(Debug, Copy, Clone)]
pub enum ScoreNorm {
    /// Divide by minimum graph or node degree
    MinDegree,

    /// Divide by maximum graph or node degree
    MaxDegree,
}
