#[derive(Clone)]

///A vertex is a fundamental unit of a graph that represents an individual entity or point. Represents its neighbors with a adjacency list
pub struct Vertex {
    id: usize,
    edges: Vec<(usize, f64)>,
    degree: usize,
}

impl Vertex {
    /// Returns a new vertex.
    pub fn new(id: usize) -> Self {
        Vertex {
            id,
            edges: Vec::new(),
            degree: 0,
        }
    }

    fn add_degree(&mut self) {
        self.degree = self.degree + 1;
    }

    /// Adds a edge to the vertex
    pub fn add_edge(&mut self, target: &usize, weight: &f64) {
        self.add_degree();
        self.edges.push((*target, *weight));
    }

    /// Returns the id from the vertex
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the edges from the vertex
    pub fn edges(self) -> Vec<(usize, f64)> {
        self.edges
    }

    /// Returns the degree from the vertex
    pub fn degree(&self) -> usize {
        self.degree
    }
}
