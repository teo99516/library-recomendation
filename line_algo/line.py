import tensorflow as tf
import numpy as np
import argparse
from line_algo.model import LINEModel
from line_algo.utils import DBLPDataLoader
import pickle
import time
from argparse import Namespace

def main(args):
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--embedding_dim', type=int, default=128)
    #parser.add_argument('--batch_size', type=int, default=128)
    #parser.add_argument('--K', type=int, default=5) #Number of negative samples
    #parser.add_argument('--proximity', default='first-order', help='first-order or second-order')
    #parser.add_argument('--learning_rate', default=0.025)
    #parser.add_argument('--mode', default='train')
    #parser.add_argument('--num_batches', type=int, default=300000)
    #parser.add_argument('--total_graph', default=True)
    #parser.add_argument('--graph_file', default='data/co-authorship_graph.pkl')
    #args = parser.parse_args()
    
    #args = Namespace(embedding_dim=128, batch_size=128, K=5, proximity="first-order", learning_rate=0.025,
    #                 mode= "train", num_batches=1000, total_graph=True, graph_file= "line_algo/data/lib_rec.gpickle")
    #print(args)
    embeddings=train(args)
    #if args.mode == 'train':
    #    embeddings = train(args)
    #elif args.mode == 'test':
    #    test(args)

    return embeddings



def train(args):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('line_algo/data/embedding_%s.gpickle' % suffix, 'wb'))

    return data_loader.embedding_mapping(normalized_embedding)

def test(args):
    pass

if __name__ == '__main__':
    emb=main (args)