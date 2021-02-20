import argparse
from collections import defaultdict


def create_label_dict(input_file_name, version):
    labels_dict = {}
    idx = 0
    with open(input_file_name, 'r', encoding = "ISO-8859-1") as input_file:
        for line in input_file:
            split_line = line.split(' ')
            label = get_label(split_line[0], version=version)

            if label in labels_dict:
                continue
            else:
                labels_dict[label] = idx
                idx += 1
    return labels_dict

def get_label(label, version):
    if version == 6:
        label = label.split(':')[0]
        return label
    else:
        return label


def read_and_store_from_tsv(input_file_name, output_file_name, version, label_dict):

    with open(input_file_name, 'r', encoding = "ISO-8859-1") as input_file:
        with open(output_file_name, 'w', encoding = "ISO-8859-1") as output_file:
            for line in input_file:
                split_line = line.split(' ')
                label = get_label(split_line[0], version=version)
                sentence = " ".join(split_line[1:]).strip().encode('utf8')
                int_label = label_dict[label]
                output_file.write(f'{sentence}\t{int_label}')
                output_file.write('\n')
    return


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--label_version", default=None, type=int, required=True,
                        help="TREC 6/50")

    args = parser.parse_args()

    # Read input files from folder
    label_dict = create_label_dict(input_file_name=args.data_dir + 'train.txt',
                                   version=args.label_version)
    for file_split in ['train', 'test', 'dev']:
        input_file_name = args.data_dir + file_split + '.txt'
        output_file_name = args.data_dir + file_split + '.tsv'
        read_and_store_from_tsv(input_file_name=input_file_name,
                                output_file_name=output_file_name,
                                version=args.label_version,
                                label_dict=label_dict)


if __name__ == "__main__":
    main()
