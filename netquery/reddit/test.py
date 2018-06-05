from argparse import ArgumentParser

import torch
from netquery.utils import *
from netquery.reddit.data_utils_new import load_graph
from netquery.data_utils import load_queries_by_formula, load_test_queries_by_formula
from netquery.model import QueryEncoderDecoder
from netquery.train_helpers import run_train
from torch import optim


parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="/dfs/scratch0/nqe-reddit-new/gaming_graph")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--log_dir", type=str, default="/dfs/scratch0/nqe-reddit-new/gaming_graph/testing_logs")
parser.add_argument("--model_dir", type=str, default="/dfs/scratch0/nqe-reddit-new/trained_models")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--opt", type=str, default="adam")
args = parser.parse_args()

print "Loading graph data.."
graph, feature_modules = load_graph(args.data_dir, args.embed_dim, cuda=args.cuda)
out_dims = {mode:args.embed_dim for mode in graph.relations}

enc = get_encoder(args.depth, graph, out_dims, feature_modules, args.cuda)
dec = get_metapath_decoder(graph, enc.out_dims if args.depth > 0 else out_dims, args.decoder)
inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder)
    
enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
if args.cuda:
    enc_dec.cuda()

log_file = args.log_dir + "/{data:s}-{depth:d}-{embed_dim:d}-{lr:f}-{decoder:s}-{inter_decoder:s}.log".format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        inter_decoder=args.inter_decoder)
model_file = args.model_dir + "/{data:s}-{depth:d}-{embed_dim:d}-{lr:f}-{decoder:s}-{inter_decoder:s}.log".format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        inter_decoder=args.inter_decoder)
logger = setup_logging(log_file)
enc_dec.load_state_dict(torch.load(model_file))
print "Loading edge data.."
train_queries = load_queries_by_formula(args.data_dir + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(args.data_dir + "/val_edges-split.pkl")
test_queries = load_test_queries_by_formula(args.data_dir + "/test_edges-split.pkl")

print "Loading query data.."
for i in range(2,4):
    train_queries.update(load_queries_by_formula(args.data_dir + "/train_queries_{:d}.pkl".format(i)))
    i_val_queries = load_test_queries_by_formula(args.data_dir + "/val_queries_{:d}-newclean.pkl".format(i))
    val_queries["one_neg"].update(i_val_queries["one_neg"])
    val_queries["full_neg"].update(i_val_queries["full_neg"])
    i_test_queries = load_test_queries_by_formula(args.data_dir + "/test_queries_{:d}-newclean.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])


optimizer = optim.Adam(filter(lambda p : p.requires_grad, enc_dec.parameters()), lr=args.lr)
run_train(enc_dec, optimizer, train_queries, val_queries, test_queries, logger, max_iter=1)
