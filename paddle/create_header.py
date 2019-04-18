# -*- coding:utf-8 -*-
import io

from mdataset import MmapFile


def create_header_file(data_path):
    headerfile = data_path + '.header'
    with io.open(data_path, 'r', encoding='utf-8') as fin, \
            io.open(headerfile, 'w') as fout:
        start, end, size = 0, 0, 0
        for lidx, line in enumerate(fin):
            start += size
            size = len(line.encode('utf-8'))
            fout.write('{}\t{}\t{}\n'.format(lidx, start, size))


def main():
    from args import parse_args
    args = parse_args()
    print(args)
    dataset = args.trainset + args.testset + args.devset
    print(dataset)
    for data_path in dataset:
        create_header_file(data_path)
        reader = MmapFile(data_path)

        # test sample
        line = reader.getvalue('10')
        try:
            sample = bytes.decode(line.strip())
            print(sample)
        except:
            print(line)


if __name__ == '__main__':
    main()
