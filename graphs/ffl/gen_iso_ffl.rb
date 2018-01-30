

class Graph
    attr_reader :input_nodes, :hidden_nodes, :output_nodes

    class Node < Struct.new(:id, :weight); end

    def initialize(node_ids, node_weights)
	@input_nodes = []
	@hidden_nodes = []
	@output_nodes = []
	@node_ids = node_ids
	@node_weights = node_weights
	@edges = []
    end

    def create_input_node
	@input_nodes.push(Node.new(next_node_id(), next_node_weight()))
    end

    def create_output_node
	@output_nodes.push(Node.new(next_node_id(), next_node_weight()))
    end

    def create_hidden_node
	@hidden_nodes.push(Node.new(next_node_id(), next_node_weight()))
    end

    def create_edge(source_id, target_id)
	raise ArgumentError unless source_id.kind_of? Integer
	raise ArgumentError unless target_id.kind_of? Integer
	@edges.push([source_id, target_id])
    end

    private def next_node_id
	@node_ids.pop || raise
    end
    private def next_node_weight
	@node_weights.pop || raise
    end

    private def write_gml_node(node, out)
	out << "  node [id #{node.id} weight #{node.weight}]\n"
    end
    private def write_gml_edge(edge, out)
    	out << "  edge [source #{edge[0]} target #{edge[1]}]\n"
    end

    def write_gml(out="")
	out << "graph\n"
	out << "[\n"
	out << "  directed 1\n"
	@input_nodes.each {|node| write_gml_node(node, out) }
	@hidden_nodes.each {|node| write_gml_node(node, out) }
	@output_nodes.each {|node| write_gml_node(node, out) }
	@edges.each {|edge| write_gml_edge(edge, out) }
	out << "]\n"
	return out
    end
end


def create_graph(n, nodes)
    g = Graph.new(nodes.map{|node| node.id}, nodes.map{|node| node.weight})
    n.times do 
	g.create_input_node
	g.create_hidden_node
	g.create_output_node
    end 
    n.times do |i|
	g.create_edge(g.input_nodes[i].id, g.hidden_nodes[i].id)
	g.create_edge(g.hidden_nodes[i].id, g.output_nodes[i].id)
	g.create_edge(g.input_nodes[i].id, g.output_nodes[i].id)
    end 
    return g
end

if __FILE__ == $0
	N = Integer(ARGV.shift)
	nodes = (0 ... (3*N)).map {|i| Graph::Node.new(i, rand()) }

	g1 = create_graph(N, nodes.shuffle) 
	g2 = create_graph(N, nodes.shuffle)

	File.write("ffl_iso_1_#{N}.gml", g1.write_gml())
	File.write("ffl_iso_2_#{N}.gml", g2.write_gml())
end
