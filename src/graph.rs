use core::f64;
use ordered_float::OrderedFloat;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    usize,
};

use crate::vertex::Edge;
use crate::vertex::Vertex;
use crate::vertex::VertexStatus;

#[derive(Debug, Clone, Eq, PartialEq)]
struct State {
    cost: OrderedFloat<f64>,
    position: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) // inversão para que o BinaryHeap funcione como uma min-heap
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A graph is a mathematical structure used to model pairwise relationships between objects. It consists of vertices connected by edges.
pub struct Graph {
    vertices: Vec<Option<Vertex>>,
    is_bidirectional: bool,
    components: Option<Vec<Vec<usize>>>,
    has_negative_weights: bool,
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
            components: None,
            has_negative_weights: false,
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

        if *weight < 0.0 {
            self.has_negative_weights = true
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
        start_id: usize,
        stop_id: Option<usize>,
    ) -> (Vec<Option<Vertex>>, Vec<usize>, Vec<Option<usize>>) {
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut visited_order: Vec<usize> = Vec::new();

        let n = self.vertices_number();
        let mut levels: Vec<usize> = vec![0; n];
        let mut parents: Vec<Option<usize>> = vec![None; n];

        let mut stop_flag: bool = false;

        if let Some(root) = &mut self.vertices[start_id] {
            queue.push_back(start_id);
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

        (self.vertices.clone(), levels, parents)
    }

    fn dfs_core(
        &mut self,
        start_id: usize,
    ) -> (Vec<Option<Vertex>>, Vec<usize>, Vec<Option<usize>>) {
        let mut stack: Vec<usize> = Vec::new();
        let mut visited_order: Vec<usize> = Vec::new();

        let n = self.vertices_number();
        let mut levels: Vec<usize> = vec![0; n];
        let mut parents: Vec<Option<usize>> = vec![None; n];

        if let Some(root) = &mut self.vertices[start_id] {
            stack.push(start_id);
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

        (self.vertices.clone(), levels, parents)
    }
}

///Methods for various algorithms within graphs
impl Graph {
    /// Calculates the distance between the start_id and the stop_id using BFS
    pub fn calculate_distance(&mut self, start_id: usize, stop_id: usize) -> usize {
        self.unmark_all_vertices();

        let mut distance = usize::MIN;
        let data = self.bfs_core(start_id, Some(stop_id));

        for vertex_opt in data.0 {
            if let Some(vertex) = vertex_opt {
                if vertex.id() > distance {
                    distance = vertex.id();
                }
            }
        }

        distance
    }

    /// Calculates the diameter of the graph
    /// * `mode` - Use exact to compute using the default algorithm, slow on large graphs. Use approximate for when working with large graphs
    pub fn calculate_diameter(&mut self, mode: &str) -> Result<usize, &str> {
        self.unmark_all_vertices();

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
                    for i in 0..bfs.0.len() {
                        if bfs.1[i] > diameter {
                            diameter = bfs.1[i];
                            chosen_vertex = bfs.0[i].clone().unwrap().id();
                        }
                    }
                }
            }

            let bfs = self.bfs_core(chosen_vertex, None);
            for i in 0..bfs.0.len() {
                if bfs.1[i] > diameter {
                    diameter = bfs.1[i];
                }
            }

            return Ok(diameter);
        }

        Err("A method was not chosen, please choose one.")
    }

    /// Calculates and returns the connected components of the graph. It works by searching with DFS in loop using all unmarked vertices
    pub fn calculate_connected_components(&mut self) -> Vec<Vec<usize>> {
        if let Some(components) = &self.components {
            return components.to_vec();
        }
        self.unmark_all_vertices();

        let mut components: Vec<Vec<usize>> = Vec::new();
        let vertex_count = self.vertices_number();

        for vertex_idx in 0..vertex_count {
            if let Some(vertex) = &self.vertices[vertex_idx] {
                match vertex.status {
                    VertexStatus::Marked => (),
                    VertexStatus::Unmarked => {
                        let component = self
                            .dfs_core(vertex.id())
                            .0
                            .into_iter()
                            .map(|x| x.unwrap().id())
                            .collect::<Vec<usize>>();
                        components.push(component);
                    }
                }
            }
        }

        self.components = Some(components.clone());

        components
    }
}

impl Graph {
    pub fn create_djikstra_vector(
        &self,
        root_id: usize,
        destinations: &[usize],
    ) -> Result<(Vec<f64>, Vec<Vec<Edge>>), &str> {
        if self.has_negative_weights {
            return Err(
                "This library does not work with negative cycles for minimum distances computing",
            );
        }

        let n = self.vertices_number();
        let mut distances = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        let mut parents = vec![None; n];

        distances[root_id] = 0.0;

        for _ in 0..n {
            let mut u = None;
            for i in 0..n {
                if !visited[i] && (u.is_none() || distances[i] < distances[u.unwrap()]) {
                    u = Some(i);
                }
            }

            let u = match u {
                Some(v) => v,
                None => break,
            };

            visited[u] = true;

            if let Some(vertex) = &self.vertices[u] {
                for edge in &vertex.edges {
                    let new_distance = distances[u] + edge.weight;
                    if new_distance < distances[edge.target] {
                        distances[edge.target] = new_distance;
                        parents[edge.target] = Some(u);
                    }
                }
            }
        }

        let mut paths = Vec::new();
        for &target in destinations {
            let mut path = Vec::new();
            let current = target;

            while let Some(parent) = parents[current] {
                let weight = distances[current] - distances[parent];
                path.push(Edge {
                    target: current,
                    weight,
                })
            }

            path.reverse();
            paths.push(path);
        }

        Ok((distances, paths))
    }

    pub fn create_distance_heap(
        &self,
        root_id: usize,
        destinations: &[usize],
    ) -> Result<(Vec<f64>, Vec<Vec<Edge>>), &str> {
        if self.has_negative_weights {
            return Err(
                "This library does not work with negative cycles for minimum distances computing",
            );
        }

        let n = self.vertices_number();
        let mut distances = vec![OrderedFloat(f64::INFINITY); n];
        let mut heap = BinaryHeap::new();
        let mut parents: Vec<Option<usize>> = vec![None; n];

        distances[root_id] = OrderedFloat(0.0);
        heap.push(State {
            cost: OrderedFloat(0.0),
            position: root_id,
        });

        while let Some(State { cost, position }) = heap.pop() {
            if cost > distances[position] {
                continue;
            }

            if let Some(vertex) = &self.vertices[position] {
                for edge in &vertex.edges {
                    let next_cost = cost + OrderedFloat(edge.weight);

                    if next_cost < distances[edge.target] {
                        distances[edge.target] = next_cost;
                        parents[edge.target] = Some(position);
                        heap.push(State {
                            cost: next_cost,
                            position: edge.target,
                        });
                    }
                }
            }
        }

        // Converte `distances` de OrderedFloat<f64> para f64 puro
        let distances: Vec<f64> = distances.into_iter().map(|d| d.into_inner()).collect();

        // Construção dos caminhos mínimos para os vértices de `destinations`
        let mut paths = Vec::new();
        for &target in destinations {
            let mut path = Vec::new();
            let mut current = target;

            // Segue o caminho de `target` até `root_id` usando o vetor `parents`
            while let Some(parent) = parents[current] {
                let weight = distances[current] - distances[parent];
                path.push(Edge {
                    target: current,
                    weight,
                });
                current = parent;
            }

            path.reverse(); // Inverte o caminho para que vá do `root_id` ao `target`
            paths.push(path);
        }

        Ok((distances, paths))
    }
}
