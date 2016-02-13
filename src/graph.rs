use super::traits::{EdgeWeight, Edges, Graph};
use closed01::Closed01;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use petgraph::{EdgeDirection, Directed};
use petgraph::Graph as PetGraph;
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
            weight: weight,
        }
    }
}

#[derive(Debug)]
pub struct EdgeList {
    edges: Vec<Edge>,
}

impl EdgeList {
    pub fn new(edges: Vec<Edge>) -> EdgeList {
        EdgeList { edges: edges }
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
pub struct Node<T> {
    in_edges: EdgeList,
    out_edges: EdgeList,
    node_value: T,
}


impl<T: Debug> Node<T> {
    pub fn new(in_edges: EdgeList, out_edges: EdgeList, node_value: T) -> Node<T> {
        Node {
            in_edges: in_edges,
            out_edges: out_edges,
            node_value: node_value,
        }
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
pub struct OwnedGraph<T: Debug + Default> {
    nodes: Vec<Node<T>>,
}

impl<T: Debug + Default> OwnedGraph<T> {
    pub fn new(nodes: Vec<Node<T>>) -> OwnedGraph<T> {
        OwnedGraph { nodes: nodes }
    }

    pub fn from_petgraph(pg: &PetGraph<(), (), Directed>) -> OwnedGraph<()> {
        OwnedGraph {
            nodes: pg.node_indices()
                     .map(|i| {
                         Node::new(EdgeList::new(pg.edges_directed(i, EdgeDirection::Incoming)
                                                   .map(|(j, _w)| Edge::new_unweighted(j.index()))
                                                   .collect()),
                                   EdgeList::new(pg.edges_directed(i, EdgeDirection::Outgoing)
                                                   .map(|(j, _w)| Edge::new_unweighted(j.index()))
                                                   .collect()),
                                   ())
                     })
                     .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn push_empty_node(&mut self) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new(EdgeList::new(Vec::new()),
                                  EdgeList::new(Vec::new()),
                                  T::default()));
        return idx;
    }
}

impl<T: Debug + Default> Graph for OwnedGraph<T> {
    type E = EdgeList;
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    fn node_degree(&self, node_idx: usize) -> usize {
        self.nodes[node_idx].degree()
    }

    #[inline]
    fn in_edges_of<'b>(&'b self, node_idx: usize) -> &'b Self::E {
        &&self.nodes[node_idx].in_edges
    }

    #[inline]
    fn out_edges_of<'b>(&'b self, node_idx: usize) -> &'b Self::E {
        &self.nodes[node_idx].out_edges
    }
}

pub struct GraphBuilder<T: Debug + Default> {
    // maps node_id to index in node_in_edges/node_out_edges.
    node_map: BTreeMap<usize, usize>,
    graph: OwnedGraph<T>,
}

impl<T: Debug + Default> GraphBuilder<T> {
    pub fn new() -> GraphBuilder<T> {
        GraphBuilder {
            node_map: BTreeMap::new(),
            graph: OwnedGraph::new(Vec::new()),
        }
    }

    pub fn graph(self) -> OwnedGraph<T> {
        self.graph
    }

    // XXX:Add a node.

    // returns node index
    pub fn add_or_replace_node(&mut self, node_id: usize) -> usize {
        match self.node_map.entry(node_id) {
            Entry::Vacant(e) => {
                let next_id = self.graph.push_empty_node();
                e.insert(next_id);
                return next_id;
            }
            Entry::Occupied(e) => {
                // XXX: replace?
                return *e.get();
            }
        }
    }

    pub fn add_edge_unweighted(&mut self, source_node_id: usize, target_node_id: usize) {
        self.add_edge(source_node_id, target_node_id, Closed01::zero());
    }

    pub fn add_edge(&mut self, source_node_id: usize, target_node_id: usize, weight: EdgeWeight) {
        let source_index = self.add_or_replace_node(source_node_id);
        let target_index = self.add_or_replace_node(target_node_id);
        // let edge_out = Edge::new(target_index)
        self.graph.nodes[source_index].add_out_edge(Edge::new(target_index, weight));
        self.graph.nodes[target_index].add_in_edge(Edge::new(source_index, weight));
    }
}
