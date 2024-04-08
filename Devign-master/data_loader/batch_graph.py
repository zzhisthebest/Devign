import torch
from dgl import DGLGraph


class BatchGraph:
    def __init__(self):
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)
        num_new_nodes = _g.number_of_nodes()
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes)))
        self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        sources, dests = _g.all_edges()
        sources += self.number_of_nodes
        dests += self.number_of_nodes
        self.graph.add_edges(sources, dests, data=_g.edata)
        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1

    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    # 将批量化的图数据解包回原始的单个图数据。
    #具体来说，它从批量化的特征中提取每个单独图的特征，并将它们重新组织为单个图的特征集合。
    def de_batchify_graphs(self, features=None):#己。
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]
        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat(
        (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,device=v.device)),
                dim=0
            )
        output_vectors = torch.stack(vectors)
        lengths = torch.LongTensor(lengths).to(device=output_vectors.device)
        return output_vectors, lengths

    def get_network_inputs(self, cuda=False):#己。
        raise NotImplementedError('Must be implemented by subclasses.')


class GGNNBatchGraph(BatchGraph):#己。
    def get_network_inputs(self, cuda=False, device=None):
        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return self.graph, features, edge_types
        pass
