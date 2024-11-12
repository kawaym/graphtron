use std::collections::VecDeque;

use crate::vertex;
use crate::vertex::Vertex;
use crate::vertex::VertexStatus;

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

impl Graph {
    fn unmark_all_vertices(&mut self) {
        for vertex_opt in &mut self.vertices {
            if let Some(vertex) = vertex_opt {
                vertex.unmark();
            }
        }
    }

    fn bfs_core(
        &mut self,
        root_id: usize,
        stop_id: Option<usize>,
    ) -> Vec<(usize, usize, Option<usize>)> {
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut visited_order: Vec<usize> = Vec::new();
        let mut tree: Vec<(usize, usize, Option<usize>)> = Vec::new();

        let n = self.vertices_number();
        let mut levels: Vec<usize> = vec![0; n];
        let mut parents: Vec<Option<usize>> = vec![None; n];

        let mut stop_flag: bool = false;

        self.unmark_all_vertices();

        if let Some(root) = &mut self.vertices[root_id] {
            queue.push_back(root_id);
            root.mark();

            loop {
                if let Some(id) = queue.pop_front() {
                    visited_order.push(id);
                    if let Some(parent) = parents[id] {
                        levels[id] = levels[parent] + 1
                    }

                    if stop_flag {
                        break;
                    }

                    if let Some(vertex) = &self.vertices[id].clone() {
                        for edge in &vertex.edges {
                            match self.vertices[edge.target].as_mut().unwrap().status {
                                VertexStatus::Marked => (),
                                VertexStatus::Unmarked => {
                                    parents[edge.target] = Some(id);
                                    self.vertices[edge.target].as_mut().unwrap().mark();
                                    queue.push_back(edge.target);
                                }
                            }
                            if let Some(stop_id_parsed) = stop_id {
                                if stop_id_parsed == edge.target {
                                    stop_flag = true;
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }

        // returns the node, the level of the node and the parent of the node

        for i in 0..visited_order.len() {
            tree.push((i, levels[i], parents[i]))
        }

        tree
    }

    fn dfs_core(&mut self, root_id: usize) -> Vec<(usize, usize, Option<usize>)> {
        let mut stack: Vec<usize> = Vec::new();
        let mut visited_order: Vec<usize> = Vec::new();
        let mut tree: Vec<(usize, usize, Option<usize>)> = Vec::new();

        let n = self.vertices_number();
        let mut levels: Vec<usize> = vec![0; n];
        let mut parents: Vec<Option<usize>> = vec![None; n];

        self.unmark_all_vertices();

        if let Some(root) = &mut self.vertices[root_id] {
            stack.push(root_id);
            root.mark();

            loop {
                if let Some(id) = stack.pop() {
                    visited_order.push(id);
                    if let Some(parent) = parents[id] {
                        levels[id] = levels[parent] + 1
                    }

                    if let Some(vertex) = &self.vertices[id].clone() {
                        for edge in &vertex.edges {
                            match self.vertices[edge.target].as_mut().unwrap().status {
                                VertexStatus::Marked => (),
                                VertexStatus::Unmarked => {
                                    parents[edge.target] = Some(id);
                                    self.vertices[edge.target].as_mut().unwrap().mark();
                                    stack.push(edge.target);
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }

        // returns the node, the level of the node and the parent of the node

        for i in 0..visited_order.len() {
            tree.push((i, levels[i], parents[i]))
        }

        tree
    }
}

///Methods for various algorithms within graphs
impl Graph {
    /// Calculates the distance between the root_id and the stop_id using BFS
    pub fn calculate_distance(&mut self, root_id: usize, stop_id: usize) -> usize {
        let mut distance = usize::MIN;
        let data = self.bfs_core(root_id, Some(stop_id));

        for vector in data {
            if vector.1 > distance {
                distance = vector.1;
            }
        }

        distance
    }

    /// Calculates the diameter of the graph
    /// * `mode` - Use exact to compute using the default algorithm, slow on large graphs. Use approximate for when working with large graphs
    pub fn calculate_diameter(&mut self, mode: &str) -> Result<usize, &str> {
        //TODO encontrar maneira de iterar sobre os vértices sem utilizar clone para evitar carga na memória
        let mut diameter = usize::MIN;

        if mode == "exact" {
            for i in 0..self.vertices.len() {
                if let Some(vertex_start) = &self.vertices[i].clone() {
                    for j in 0..self.vertices.len() {
                        if let Some(vertex_end) = &self.vertices[j].clone() {
                            if vertex_start.id() == vertex_end.id() {
                                continue;
                            }

                            let distance =
                                &mut self.calculate_distance(vertex_start.id(), vertex_end.id());
                            if *distance > diameter {
                                diameter = *distance;
                            }
                        }
                    }
                }
            }
            return Ok(diameter);
        }

        if mode == "approximate" {
            let mut chosen_vertex: usize = 0;
            let vertex_count = self.vertices_number();

            for vertex_idx in 0..vertex_count {
                if let Some(vertex) = &self.vertices[vertex_idx] {
                    let bfs = self.bfs_core(vertex.id(), None);
                    for i in bfs {
                        if i.1 > diameter {
                            diameter = i.1;
                            chosen_vertex = i.0;
                        }
                    }
                }
            }

            let bfs = self.bfs_core(chosen_vertex, None);
            for i in bfs {
                if i.1 > diameter {
                    diameter = i.1;
                    chosen_vertex = i.0;
                }
            }

            return Ok(diameter);
        }

        Err("A method was not chosen, please choose one.")
    }
}
