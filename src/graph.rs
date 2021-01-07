use crate::traits::{EdgeWeight, Edges, Graph};
use closed01::Closed01;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph as PetGraph;
use petgraph::{Directed, EdgeDirection};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Edge {
    /// Node index type. Our graphs never exceed 4 billion nodes.
    pointing_node: u32,
    weight: EdgeWeight,
}

impl Edge {
    pub fn new_unweighted(node_idx: usize) -> Edge {
        Edge::new(node_idx, Closed01::zero())
    }

    pub fn new(node_idx: usize, weight: EdgeWeight) -> Edge {
        assert!(node_idx <= u32::max_value() as usize);
        Edge {
            pointing_node: node_idx as u32,
            weight,
        }
    }
}

#[derive(Debug)]
pub struct EdgeList {
    edges: Vec<Edge>,
}

impl EdgeList {
    pub fn new(edges: Vec<Edge>) -> EdgeList {
        EdgeList { edges }
    }
}

impl Edges for EdgeList {
    #[inline]
    fn num_edges(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    fn nth_edge(&self, n: usize) -> Option<usize> {
        self.edges.get(n).map(|n| n.pointing_node as usize)
    }

    #[inline]
    fn nth_edge_weight(&self, n: usize) -> Option<EdgeWeight> {
        self.edges.get(n).map(|n| n.weight)
    }
}

#[derive(Debug)]
pub struct Node<T: Debug + Clone> {
    in_edges: EdgeList,
    out_edges: EdgeList,
    node_value: T,
}

impl<T: Debug + Clone> Node<T> {
    pub fn new(in_edges: EdgeList, out_edges: EdgeList, node_value: T) -> Node<T> {
        Node {
            in_edges,
            out_edges,
            node_value,
        }
    }

    pub fn node_value(&self) -> &T {
        &self.node_value
    }

    pub fn add_in_edge(&mut self, edge: Edge) {
        self.in_edges.edges.push(edge);
    }

    pub fn add_out_edge(&mut self, edge: Edge) {
        self.out_edges.edges.push(edge);
    }

    fn degree(&self) -> usize {
        self.in_edges.num_edges() + self.out_edges.num_edges()
    }
}

#[derive(Debug)]
pub struct OwnedGraph<T: Debug + Clone> {
    nodes: Vec<Node<T>>,
}

impl<T: Debug + Clone> OwnedGraph<T> {
    pub fn new(nodes: Vec<Node<T>>) -> OwnedGraph<T> {
        OwnedGraph { nodes }
    }

    pub fn nodes(&self) -> &[Node<T>] {
        &self.nodes
    }

    pub fn from_petgraph(pg: &PetGraph<T, EdgeWeight, Directed>) -> OwnedGraph<T> {
        OwnedGraph {
            nodes: pg
                .node_indices()
                .map(|i| {
                    Node::new(
                        EdgeList::new(
                            pg.edges_directed(i, EdgeDirection::Incoming)
                                .map(|edge| Edge::new(edge.source().index(), *edge.weight()))
                                .collect(),
                        ),
                        EdgeList::new(
                            pg.edges_directed(i, EdgeDirection::Outgoing)
                                .map(|edge| Edge::new(edge.target().index(), *edge.weight()))
                                .collect(),
                        ),
                        pg.node_weight(i).unwrap().clone(),
                    )
                })
                .collect(),
        }
    }

    pub fn to_petgraph(&self) -> PetGraph<T, EdgeWeight, Directed> {
        let mut graph = PetGraph::new();

        for node in self.nodes() {
            graph.add_node(node.node_value().clone());
        }

        for (source_idx, node) in self.nodes().iter().enumerate() {
            for edge in &node.out_edges.edges {
                let target_idx = edge.pointing_node as usize;
                graph.add_edge(
                    NodeIndex::new(source_idx),
                    NodeIndex::new(target_idx),
                    edge.weight,
                );
            }
        }

        graph
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push_empty_node(&mut self, node_value: T) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new(
            EdgeList::new(Vec::new()),
            EdgeList::new(Vec::new()),
            node_value,
        ));
        idx
    }
}

impl<T: Debug + Clone> Graph for OwnedGraph<T> {
    type EDGE = EdgeList;
    type NODE = T;
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    fn node_value(&self, node_idx: usize) -> &Self::NODE {
        self.nodes[node_idx].node_value()
    }

    #[inline]
    fn node_degree(&self, node_idx: usize) -> usize {
        self.nodes[node_idx].degree()
    }

    #[inline]
    fn in_edges_of(&self, node_idx: usize) -> &Self::EDGE {
        &&self.nodes[node_idx].in_edges
    }

    #[inline]
    fn out_edges_of(&self, node_idx: usize) -> &Self::EDGE {
        &self.nodes[node_idx].out_edges
    }
}

pub struct GraphBuilder<K: Ord, T: Debug + Clone> {
    // maps node_id to index in node_in_edges/node_out_edges.
    node_map: BTreeMap<K, usize>,
    graph: OwnedGraph<T>,
}

impl<K: Ord, T: Debug + Clone> Default for GraphBuilder<K, T> {
    fn default() -> Self {
        Self {
            node_map: BTreeMap::new(),
            graph: OwnedGraph::new(Vec::new()),
        }
    }
}

impl<K: Ord, T: Debug + Clone> GraphBuilder<K, T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn graph(self) -> OwnedGraph<T> {
        self.graph
    }

    /// Panics if `node_id` already exists.
    pub fn add_node(&mut self, node_id: K, node_value: T) -> usize {
        match self.node_map.entry(node_id) {
            Entry::Vacant(e) => {
                let next_id = self.graph.push_empty_node(node_value);
                e.insert(next_id);
                next_id
            }
            Entry::Occupied(_) => {
                panic!("Node already exists");
            }
        }
    }

    pub fn add_edge_unweighted(&mut self, source_node_id: K, target_node_id: K) {
        self.add_edge(source_node_id, target_node_id, Closed01::zero());
    }

    pub fn add_edge(&mut self, source_node_id: K, target_node_id: K, weight: EdgeWeight) {
        let source_index = *self.node_map.get(&source_node_id).unwrap();
        let target_index = *self.node_map.get(&target_node_id).unwrap();
        self.graph.nodes[source_index].add_out_edge(Edge::new(target_index, weight));
        self.graph.nodes[target_index].add_in_edge(Edge::new(source_index, weight));
    }
}
