import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as f
from torch.nn import functional as F

class DevignModel(nn.Module):#己。完全符合论文，除了把ggrn改成了ggnn
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):#num_steps是什么？GatedGraphConv的一个参数
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        #这一段看懂了
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)
        # 这一段看懂了
        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)
        # 这一段看懂了
        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)#获取这一批的数据
        # print("graph",graph)
        # print("features",features)
        # print("edge_types",edge_types)
        outputs = self.ggnn(graph, features, edge_types)
        x_i, _ = batch.de_batchify_graphs(features)#形状是
        print("x_i.shape",x_i.shape)
        h_i, _ = batch.de_batchify_graphs(outputs)#形状是
        print("h_i.shape",h_i.shape)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )#Y的第一个卷积层输出
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)#Y的第二个卷积层输出
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )#Z的第一个卷积层输出
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)#Z的第二个卷积层输出
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result

##这就是相对于devign的传统的做法，只是这里的MLP只有一层。同时这也是reveal预训练ggnn的模型
#即reveal抛弃了devign的conv层，而仅仅使用ggnn部分。
class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):#num_steps是ggnn的层数
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        #self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        # 修改输出维度为46
        self.classifier = nn.Linear(in_features=output_dim, out_features=46)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to('cuda:0')
        features = features.to('cuda:0')
        #edge_types =edge_types.to('cuda:0')
        #print("features.shape",features.shape)
        outputs = self.ggnn(graph, features, edge_types)#形状为[这一批的图的结点数之和,graph_embed_size]
        #print("outputs.shape",outputs.shape)#[一个会变的数字，200]，即[,200]
        h_i, _ = batch.de_batchify_graphs(outputs)
        #print("h_i",h_i)
        #print("h_i.shape",h_i.shape)#[128,一个会变的数字,200]，即[batch_size,这一批的图最大的结点数(结点数没这么多的图填充0),graph_embed_size]
        # 应用tanh激活函数
        h_i = F.tanh(h_i)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        #result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        result = ggnn_sum#还是为了多分类，这里不用softmax是因为交叉熵损失函数自带softmax
        return result,h_i#result的形状是torch.Size([128])即batch_size