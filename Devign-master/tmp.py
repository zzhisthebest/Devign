import argparse
import os
import pickle
import sys
import json
import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train
from utils import tally_param, debug

def get_ggnn_output(model,num_batches, data_iter):
    ggnn_output = []
    with torch.no_grad():#不加这个就会保存所有批的梯度而不释放，耗费大量的gpu内存，导致cuda out of memory
        for j in range(num_batches):
            print(j)
            graph, targets = data_iter()
            _, graph_features = model(graph, cuda=True)
            graph_features = graph_features.detach().cpu()
            for i in range(len(targets)):
                graph_feature=graph_features[i]
                #print("graph_feature.shape",graph_feature.shape)
                #print(len(torch.sum(graph_feature, dim=0).tolist()))
                target=targets[i]
                #print("target",target)
                data_point = {
                    'graph_feature': torch.sum(graph_feature, dim=0).tolist(),#使用简单的sum作为aggregation函数，符合论文
                    'target': int(target.item())
                }
                ggnn_output.append(data_point)
    return ggnn_output
if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='ggnn')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.',default='devign')
    parser.add_argument('--input_dir', type=str,help='Input Directory of the parser',default='./devign_dataset/')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=169)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)
    args = parser.parse_args()

    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v0.json'),
                          valid_src=os.path.join(input_dir, 'devign-valid-v0.json'),
                          test_src=os.path.join(input_dir, 'devign-test-v0.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()#生成.bin文件
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    # 加载保存的模型状态字典
    model_state_dict = torch.load(r'models/devign/GGNNSumModel-model.bin')
    # 将状态字典加载到模型中
    model.load_state_dict(model_state_dict)
    model.eval()
    #print(len(dataset.train_batches))
    train_ggnn_output=get_ggnn_output(model, dataset.initialize_train_batch(),dataset.get_next_train_batch)
    valid_ggnn_output=get_ggnn_output(model, dataset.initialize_valid_batch(), dataset.get_next_valid_batch)
    test_ggnn_output=get_ggnn_output(model, dataset.initialize_test_batch(), dataset.get_next_test_batch)
    output_file=open('train_ggnn_output.json','w')
    json.dump(train_ggnn_output, output_file)
    output_file.close()
    output_file = open('valid_ggnn_output.json', 'w')
    json.dump(valid_ggnn_output, output_file)
    output_file.close()
    output_file = open('test_ggnn_output.json', 'w')
    json.dump(test_ggnn_output, output_file)
    output_file.close()