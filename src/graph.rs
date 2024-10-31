use std::usize;

use crate::vertex::Vertex;

/// A graph is a mathematical structure used to model pairwise relationships between objects. It consists of vertices connected by edges.
pub struct Graph {
    vertices: Vec<Option<Vertex>>,
    is_bidirectional: bool,
}

///Methods working with creation and update to the graph
impl Graph {
    /// Returns a new Graph
    pub fn new(size: usize, is_bidirectional: Option<bool>) -> Self {
        let graph = Graph {
            vertices: vec![None; size],
            is_bidirectional: match is_bidirectional {
                Some(value) => value,
                None => true,
            },
        };

        graph
    }

    /// Adds a vertex to the graph
    pub fn add_vertex(&mut self, id: &usize) {
        match &self.vertices[*id] {
            Some(_) => (),
            None => {
                self.vertices[*id] = Some(Vertex::new(*id));
            }
        }
    }

    /// Adds an edge to the graph. Returns an Error if either of the vertices does not exist
    pub fn add_edge(&mut self, from: &usize, to: &usize, weight: &f64) -> Result<(), &str> {
        match &self.vertices[*from] {
            Some(value) => value,
            None => return Err("From vertex does not exist!"),
        };

        match &self.vertices[*to] {
            Some(value) => value,
            None => return Err("To vertex does not exist!"),
        };

        let from_vertex = self.vertices[*from].as_mut().unwrap();

        from_vertex.add_edge(to, weight);

        if self.is_bidirectional {
            let to_vertex = self.vertices[*to].as_mut().unwrap();
            to_vertex.add_edge(from, weight);
        }

        Ok(())
    }
}

/// Methods working with metrics about the graph or vertices
impl Graph {
    /// Returns the number of vertices in the graph
    pub fn vertices_number(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of edges in the graph
    pub fn edges_number(&self) -> usize {
        let mut number = 0;

        for vertex in &self.vertices {
            match vertex {
                Some(value) => number += value.degree(),
                None => (),
            }
        }

        if self.is_bidirectional {
            number = number / 2
        }

        number
    }

    /// Returns the minimum degree from a vertex in the graph
    pub fn minimum_degree(&self) -> usize {
        let mut number = usize::MAX;

        for vertex_opt in &self.vertices {
            if let Some(vertex) = vertex_opt {
                if vertex.degree() < number {
                    number = vertex.degree()
                }
            }
        }

        number
    }

    /// Returns the maximum degree from a vertex in the graph
    pub fn maximum_degree(&self) -> usize {
        let mut number = usize::MIN;

        for vertex_opt in &self.vertices {
            if let Some(vertex) = vertex_opt {
                if vertex.degree() > number {
                    number = vertex.degree()
                }
            }
        }

        number
    }

    /// Returns the average degree from a vertex in the graph
    pub fn average_degree(&self) -> f64 {
        let mut number = 0.0;
        for vertex_opt in &self.vertices {
            if let Some(vertex) = vertex_opt {
                number += vertex.degree() as f64
            }
        }
        number /= self.vertices.len() as f64;

        number
    }

    /// Returns the median degree from a vertex in the graph
    pub fn median_degree(&self) -> f64 {
        let mut vertices: Vec<usize> = vec![];
        let number: f64;
        for vertex in &self.vertices {
            match vertex {
                Some(info) => {
                    vertices.push(info.degree().clone());
                }
                None => (),
            }
        }
        vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if vertices.len() % 2 != 0 {
            number = vertices[vertices.len() / 2] as f64
        } else {
            number = (vertices[vertices.len() / 2] as f64
                + vertices[vertices.len() / 2 - 1] as f64)
                / 2.0;
        }

        number
    }
}
