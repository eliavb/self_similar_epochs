import itertools

from gensim.models import Word2Vec
import numpy as np
import argparse
import networkx as nx

parser = argparse.ArgumentParser(description=('Code used to create the paper: '
                                              '"Self-similar Epochs: Value in arrangement", ICML 2019'))
parser.add_argument('--input_graph', type=str, help='input graph path to embed.')
parser.add_argument('--output_path', type=str, help='output embedding path.')
parser.add_argument('--epochs', type=int, default=10, help='number epochs to train.')
parser.add_argument('--sampling_method', type=str, help='Sampling procedure.')
parser.add_argument('--test_size', type=float, default=20, help='Test set size in percentage.')


def embed_nodes(input_graph, sampler):
    model = Word2Vec(min_count=0)
    vocab = []
    for (u, v) in input_graph.edges:
        vocab.append([u, v])
    model.build_vocab(vocab)
    model.train(sampler.get_epoch_examples(), total_examples=sampler.get_total_examples(), epochs=1)
    return model


def write_embeddings(node_embeddings, output_path):
    node_embeddings.save(output_path)


class CooSampling:

    def __init__(self, input_graph, epochs):
        self._epochs = epochs
        self._n = len(input_graph.edges)
        attributes = nx.get_node_attributes(graph, 'pos')
        self.row_nodes = [k for k, v in attributes.items() if v == 'row']
        self.col_nodes = [k for k, v in attributes.items() if v == 'col']
        self._matrix = np.zeros(shape=(len(self.row_nodes), len(self.col_nodes)))
        for row, col in itertools.product(self.row_nodes, self.col_nodes):
            row_idx = self.row_nodes.index(row)
            col_idx = self.col_nodes.index(col)
            edge_data = input_graph.get_edge_data(row, col)
            if edge_data is not None:
                self._matrix[row_idx, col_idx] = edge_data['weight']
        self._row_size, self._col_size = self._matrix.shape
        self._row_indxs = range(self._row_size)
        self._col_indxs = range(self._col_size)
        self._epoch_size = len(input_graph.edges)
        self._row_max = np.amax(self._matrix, axis=1)
        self._row_prob = self._row_max / np.sum(self._row_max)
        self._col_max = np.amax(self._matrix, axis=0)
        self._col_prob = self._col_max / np.sum(self._col_max)

    def get_total_examples(self):
        return self._n * self._epochs

    def get_epoch_examples(self):
        examples = 0
        row_batch = True
        while examples < self._epoch_size:
            batch = []
            u = np.random.uniform(0, 1)
            if row_batch:
                col = np.random.choice(self._col_indxs, p=self._col_prob)
                for row in self._row_indxs:
                    if self._matrix[row, col] > u * self._col_max[col]:
                        yield self.row_nodes[row], self.col_nodes[col]
            else:
                row = np.random.choice(self._row_indxs, p=self._row_prob)
                for col in self._col_indxs:
                    if self._matrix[row, col] > u * self._row_max[row]:
                        yield self.row_nodes[row], self.col_nodes[col]
            examples += 1
            row_batch = not row_batch
        return examples


class IndSampling:

    def __init__(self, input_graph, epochs):
        self._pairs = list(input_graph.edges)
        self._pairs_idx = list(range(len(self._pairs)))
        self._probabilities = np.array([float(input_graph.get_edge_data(u, v)['weight']) for (u, v) in self._pairs])
        self._probabilities /= np.sum(self._probabilities)
        self._n = len(input_graph.edges)
        self._epochs = epochs

    def get_total_examples(self):
        return self._n * self._epochs

    def get_epoch_examples(self):
        for _ in range(self._epochs):
            indxs = np.random.choice(self._pairs_idx, self._n, p=self._probabilities)
            for idx in indxs:
                yield self._pairs[idx]


def read_graph(input_path):
    return nx.read_gpickle(input_path)


def split_graph_to_train_and_test(input_graph, percentage_for_test, min_edges_from_sample=20):
    test_edges = []
    size_before_test = len(input_graph.edges)
    test_size = int(size_before_test * percentage_for_test / 100)
    for n in input_graph.nodes:
        nbrs = list(input_graph.neighbors(n))
        if len(nbrs) > min_edges_from_sample:
            np.random.shuffle(nbrs)
            test_edges += [(nbr, n) for nbr in nbrs[:10] if input_graph.degree(nbr) > 1]
        if len(test_edges) > test_size:
            break
    if len(test_edges) > test_size:
        test_edges = test_edges[:test_size]
    input_graph.remove_edges_from(test_edges)
    return test_edges, input_graph


if __name__ == '__main__':
    args = parser.parse_args()
    graph = read_graph(args.input_graph)
    _, train_graph = split_graph_to_train_and_test(graph, args.test_size)
    if args.sampling_method == 'coo':
        sampler = CooSampling(graph, args.epochs)
    if args.sampling_method == 'ind':
        sampler = IndSampling(graph, args.epochs)
    node_embedding = embed_nodes(graph, sampler)
    write_embeddings(node_embedding, args.output_path)
