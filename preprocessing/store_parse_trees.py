import argparse
import csv
import json
from typing import Dict

from constituency_parse import ParseTree


class ParsedDataset(object):
    def __init__(self, tokenizer_name):
        self.parse_trees: Dict[str, str] = {}
        self.parser = ParseTree(tokenizer_name=tokenizer_name)

    def read_and_store_from_tsv(self, input_file_name, output_file_name):
        with open(output_file_name, 'w') as output_file:
            with open(input_file_name, 'r') as open_file:
                reader = csv.reader(open_file, delimiter='\t')
                next(reader, None)  # skip header
                for row in reader:
                    text = row[0]
                    parse_tree, nt_idx_matrix = self.parser.get_parse_tree_for_raw_sent(raw_sent=text)
                    datapoint_dict = {'sentence': row[0],
                                      'parse_tree': parse_tree,
                                      'label': row[1],
                                      'nt_idx_matrix': nt_idx_matrix}
                    json.dump(datapoint_dict, output_file)
                    output_file.write('\n')
        return

    def store_parse_trees(self, output_file):
        with open(output_file, 'w') as open_file:
            json.dump(self.parse_trees, open_file)
        return


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, required=True,
                        help="Tokenizer name")

    args = parser.parse_args()
    parsed_data = ParsedDataset(tokenizer_name=args.tokenizer_name)

    # Read input files from folder
    for file_split in ['train','dev']:
        input_file_name = args.data_dir + "/" + file_split + '.tsv'
        output_file_name = args.data_dir + "/" + file_split + '_with_parse.json'
        parsed_data.read_and_store_from_tsv(input_file_name=input_file_name,
                                            output_file_name=output_file_name)


if __name__ == "__main__":
    main()
