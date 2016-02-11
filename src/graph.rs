use super::traits::{Edges, Graph};
use closed01::Closed01;

#[derive(Debug)]
pub struct Edge {
    /// Node index type. Our graphs never exceed 4 billion nodes.
    pointing_node: u32,
    weight: Closed01<f32>,
}

impl Edge {
    pub fn new_unweighted(node_idx: usize) -> Edge {
        Edge::new(node_idx, Closed01::zero())
    }

    pub fn new(node_idx: usize, weight: Closed01<f32>) -> Edge {
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
    fn nth_edge_weight(&self, n: usize) -> Option<Closed01<f32>> {
        self.edges.get(n).map(|n| n.weight)
    }
}

#[derive(Debug)]
pub struct Node {
    in_edges: EdgeList,
    out_edges: EdgeList,
}


impl Node {
    pub fn new(in_edges: EdgeList, out_edges: EdgeList) -> Node {
        Node {
            in_edges: in_edges,
            out_edges: out_edges,
        }
    }

    fn degree(&self) -> usize {
        self.in_edges.num_edges() + self.out_edges.num_edges()
    }
}

#[derive(Debug)]
pub struct OwnedGraph {
    nodes: Vec<Node>,
}
impl OwnedGraph {
    pub fn new(nodes: Vec<Node>) -> OwnedGraph {
        OwnedGraph { nodes: nodes }
    }
}

impl Graph for OwnedGraph {
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
