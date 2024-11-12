#[derive(Clone)]
pub(super) struct Edge {
    pub(super) target: usize,
    #[allow(dead_code)]
    weight: f64,
}

#[derive(Clone)]
pub(super) enum VertexStatus {
    Marked,
    Unmarked,
}

#[derive(Clone)]
///A vertex is a fundamental unit of a graph that represents an individual entity or point. Represents its neighbors with a adjacency list
pub struct Vertex {
    id: usize,
    pub(super) edges: Vec<Edge>,
    degree: usize,
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
    pub fn add_edge(&mut self, target: &usize, weight: &f64) {
        self.add_degree();
        self.edges.push(Edge {
            target: *target,
            weight: *weight,
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
