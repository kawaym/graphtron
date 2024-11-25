use core::f64;
use ordered_float::OrderedFloat;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
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

#[derive(Debug)]
/// A graph is a mathematical structure used to model pairwise relationships between objects. It consists of vertices connected by edges.
pub struct Graph {
    vertices: Vec<Option<Vertex>>,
    is_bidirectional: bool,
    components: Option<Vec<Vec<usize>>>,
    has_negative_weights: bool,
}
fn read_lines_from_file<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn read_numbers_from_line(line: String) -> Option<(String, String, String)> {
    let parts: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();

    if let (Ok(num1), Ok(num2), Ok(num3)) = (
        parts[0].parse::<String>(),
        parts[1].parse::<String>(),
        parts[2].parse::<String>(),
    ) {
        return Some((num1, num2, num3));
    }

    None
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

    pub fn clean_flow(&mut self) {
        for vert in &mut self.vertices {
            if let Some(v) = vert {
                for edge in &mut v.edges {
                    edge.flow = None;
                }
            }
        }
    }

    pub fn _read_graph(filename: &str, is_bidirectional: bool) -> Graph {
        let mut graph: Graph = Graph::new(0, Some(is_bidirectional));

        if let Ok(lines) = read_lines_from_file(filename) {
            let mut count = 0;
            for line in lines.flatten() {
                if count == 0 {
                    graph = Graph::new(line.parse::<usize>().unwrap(), Some(is_bidirectional));
                    count += 1;
                    continue;
                }
                let numbers = read_numbers_from_line(line).unwrap();
                graph.add_vertex(&(numbers.0.parse::<usize>().unwrap() - 1));
                graph.add_vertex(&(numbers.1.parse::<usize>().unwrap() - 1));
                graph.add_edge(
                    &(numbers.0.parse::<usize>().unwrap() - 1),
                    &(numbers.1.parse::<usize>().unwrap() - 1),
                    &1.0,
                    &Some(numbers.2.parse::<f64>().unwrap()),
                    &None,
                );
            }
        }
        return graph;
    }

    pub fn from_file_directional(filename: &str) -> Graph {
        Graph::_read_graph(filename, true)
    }

    pub fn from_file(filename: &str) -> Graph {
        Graph::_read_graph(filename, false)
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
    pub fn add_edge(
        &mut self,
        from: &usize,
        to: &usize,
        weight: &f64,
        capactiy: &Option<f64>,
        flow: &Option<f64>,
    ) -> Result<(), &str> {
        match &self.vertices[*from] {
            Some(value) => value,
            None => return Err("From vertex does not exist!"),
        };

        match &self.vertices[*to] {
            Some(value) => value,
            None => return Err("To vertex does not exist!"),
        };

        let from_vertex = self.vertices[*from].as_mut().unwrap();

        from_vertex.add_edge(to, weight, capactiy);

        if self.is_bidirectional {
            let to_vertex = self.vertices[*to].as_mut().unwrap();
            to_vertex.add_edge(from, weight, capactiy);
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
                    capacity: None,
                    flow: None,
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
                    capacity: None,
                    flow: None,
                });
                current = parent;
            }

            path.reverse(); // Inverte o caminho para que vá do `root_id` ao `target`
            paths.push(path);
        }

        Ok((distances, paths))
    }
}

impl Graph {
    fn find_augmenting_path(&mut self, source: usize, sink: usize) -> Option<(Vec<usize>, f64)> {
        let (_, visited_order, parents) = self.bfs_core(source, Some(sink));

        // Se o `sink` não foi visitado, não há caminho de aumento
        if parents[sink].is_none() {
            return None;
        }

        // Reconstrói o caminho de aumento a partir dos pais
        let mut path = Vec::new();
        let mut current = sink;
        while let Some(parent) = parents[current] {
            path.push(current);
            current = parent;
        }
        path.push(source);
        path.reverse();

        // Determina o fluxo máximo permitido pelo caminho
        let mut path_flow = f64::MAX;
        for i in 0..path.len() - 1 {
            let u = path[i];
            let v = path[i + 1];
            if let Some(vertex) = &self.vertices[u] {
                if let Some(edge) = vertex.edges.iter().find(|e| e.target == v) {
                    path_flow = path_flow.min(edge.residual_capacity());
                }
            }
        }

        Some((path, path_flow))
    }

    fn find_augmenting_path_ff(
        &self,
        source: usize,
        sink: usize,
        visited: &mut Vec<bool>,
        path: &mut Vec<usize>,
    ) -> Option<f64> {
        let mut stack: Vec<(usize, usize, f64)> = vec![(source, 0, f64::INFINITY)];
        let mut parents: Vec<Option<usize>> = vec![None; self.vertices.len()];

        while let Some((current, edge_index, flow)) = stack.pop() {
            // Marcar o vértice como visitado
            visited[current] = true;

            // Verificar se chegamos ao destino
            if current == sink {
                // Reconstruir o caminho usando os pais
                let mut current_node = sink;
                path.clear();
                while let Some(parent) = parents[current_node] {
                    path.push(current_node);
                    current_node = parent;
                }
                path.push(source);
                path.reverse();
                return Some(flow);
            }

            // Acessar as arestas do vértice atual
            if let Some(vertex) = &self.vertices[current] {
                for (i, edge) in vertex.edges.iter().enumerate().skip(edge_index) {
                    let residual_capacity = edge.residual_capacity();
                    if residual_capacity > 0.0 && !visited[edge.target] {
                        // Registrar o pai do vértice alvo
                        parents[edge.target] = Some(current);
                        // Adicionar o próximo estado à pilha
                        stack.push((current, i + 1, flow)); // Continuar a explorar as arestas restantes do vértice atual
                        stack.push((edge.target, 0, flow.min(residual_capacity))); // Explorar o próximo vértice
                        break;
                    }
                }
            }
        }

        None
    }

    pub fn ford_fulkerson(&mut self, source: usize, sink: usize) -> f64 {
        let mut max_flow = 0.0;

        loop {
            let mut visited = vec![false; self.vertices.len()];
            let mut path = vec![source];

            if let Some(flow) = self.find_augmenting_path_ff(source, sink, &mut visited, &mut path)
            {
                max_flow += flow;

                // Atualizar o fluxo no caminho encontrado
                for i in 0..path.len() - 1 {
                    let u = path[i];
                    let v = path[i + 1];

                    if let Some(vertex) = &mut self.vertices[u] {
                        for edge in &mut vertex.edges {
                            if edge.target == v {
                                edge.flow = Some(edge.flow.unwrap_or(0.) + flow);
                                break;
                            }
                        }
                    }

                    if let Some(vertex) = &mut self.vertices[v] {
                        for edge in &mut vertex.edges {
                            if edge.target == u {
                                edge.flow = Some(edge.flow.unwrap_or(0.) - flow);
                                break;
                            }
                        }
                    }
                }
            } else {
                break; // Nenhum caminho de aumento encontrado
            }
        }

        max_flow
    }

    fn update_flows(&mut self, path: &[usize], path_flow: f64) {
        for i in 0..path.len() - 1 {
            let u = path[i];
            let v = path[i + 1];

            // Atualiza o fluxo na aresta u -> v
            if let Some(vertex) = self.vertices[u].as_mut() {
                if let Some(edge) = vertex.edges.iter_mut().find(|e| e.target == v) {
                    edge.flow = Some(edge.flow.unwrap_or(0.) + path_flow);
                }
            }

            // Atualiza o fluxo reverso na aresta v -> u
            if let Some(vertex) = self.vertices[v].as_mut() {
                if let Some(edge) = vertex.edges.iter_mut().find(|e| e.target == u) {
                    edge.flow = Some(edge.flow.unwrap_or(0.) - path_flow);
                } else {
                    // Se a aresta reversa não existe, crie-a
                    vertex.edges.push(Edge {
                        target: u,
                        weight: 0.0,
                        capacity: Some(0.),
                        flow: Some(-path_flow),
                    });
                }
            }
        }
    }

    pub fn clone_with_zero_flows(&self) -> Graph {
        Graph {
            vertices: self
                .vertices
                .iter()
                .map(|v| {
                    v.as_ref().map(|vertex| Vertex {
                        id: vertex.id,
                        edges: vertex
                            .edges
                            .iter()
                            .map(|edge| Edge {
                                target: edge.target,
                                weight: edge.weight,
                                capacity: edge.capacity,
                                flow: Some(0.), // Inicializa fluxo como zero
                            })
                            .collect(),
                        degree: vertex.degree,
                        status: VertexStatus::Unmarked,
                    })
                })
                .collect(),
            is_bidirectional: self.is_bidirectional,
            components: None,
            has_negative_weights: self.has_negative_weights,
        }
    }

    pub fn edmonds_karp(&mut self, source: usize, sink: usize) -> f64 {
        // Cria uma cópia do grafo para operar
        let mut residual_graph = self.clone_with_zero_flows();
        let mut max_flow = 0.0;

        loop {
            // Encontra um caminho de aumento no grafo residual
            if let Some((path, path_flow)) = residual_graph.find_augmenting_path(source, sink) {
                // Atualiza o fluxo no grafo residual
                residual_graph.update_flows(&path, path_flow);

                // Incrementa o fluxo total
                max_flow += path_flow;
            } else {
                // Não há mais caminhos de aumento
                break;
            }
        }

        max_flow
    }
}
