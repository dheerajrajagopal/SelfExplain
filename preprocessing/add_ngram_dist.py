import argparse

from src.modules.prototype import ProtoVocab


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--in_file", default=None, type=str, required=True,
                        help="Input file")

    parser.add_argument("--out_file", default=None, type=str, required=True,
                        help="Output file")

    parser.add_argument("--out_vocab_file", default=None, type=str, required=True,
                        help="Output file")

    parser.add_argument("--num_labels", default=None, type=int, required=True,
                        help="Number of labels")

    args = parser.parse_args()

    # Read input files from folder
    proto_object = ProtoVocab(args.in_file, args.num_labels)
    proto_object.populate_nt_table(args.out_file, args.out_vocab_file)
    proto_object.save_dataset(args.out_file)


if __name__ == "__main__":
    main()