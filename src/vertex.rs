#[derive(Clone, Debug)]
pub struct Edge {
    pub(super) target: usize,
    pub(super) weight: f64,
    #[allow(dead_code)]
    pub(super) capacity: Option<f64>,
    #[allow(dead_code)]
    pub(super) flow: Option<f64>,
}

impl Edge {
    pub fn residual_capacity(&self) -> f64 {
        self.capacity.unwrap_or(0.) as f64 - self.flow.unwrap_or(0.) as f64
    }
}

#[derive(Clone, Debug)]
pub(super) enum VertexStatus {
    Marked,
    Unmarked,
}

#[derive(Clone, Debug)]
///A vertex is a fundamental unit of a graph that represents an individual entity or point. Represents its neighbors with a adjacency list
pub struct Vertex {
    pub(super) id: usize,
    pub(super) edges: Vec<Edge>,
    pub(super) degree: usize,
    pub(super) status: VertexStatus,
}

impl Vertex {
    /// Returns a new vertex.
    pub fn new(id: usize) -> Self {
        Vertex {
            id,
            edges: Vec::new(),
            degree: 0,
            status: VertexStatus::Unmarked,
        }
    }

    fn add_degree(&mut self) {
        self.degree = self.degree + 1;
    }

    /// Adds a edge to the vertex
    pub fn add_edge(&mut self, target: &usize, weight: &f64, capacity: &Option<f64>) {
        self.add_degree();
        self.edges.push(Edge {
            target: *target,
            weight: *weight,
            capacity: *capacity,
            flow: Some(0.),
        });
    }

    /// Returns the id from the vertex
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the degree from the vertex
    pub fn degree(&self) -> usize {
        self.degree
    }

    #[allow(dead_code)]
    pub(super) fn mark(&mut self) {
        self.status = VertexStatus::Marked
    }

    #[allow(dead_code)]
    pub(super) fn unmark(&mut self) {
        self.status = VertexStatus::Unmarked
    }
}
