#!/etc/bin/python

import json
import re
import sys
import argparse


def process(input_file,
            name_id_desc_mapping_file,
            traindata,
            trainlabel,
            testdata,
            testlabel,
            split_year):

    name_id_description_dict = dict()
    for line in name_id_desc_mapping_file:
        line = line.strip()
        name, description = line.split(':::')
        if name and description:
            name_id_description_dict[name] = description

    name_id_desc_mapping_file.close()

    assert(len(name_id_description_dict) < 27455)
    print len(name_id_description_dict)

    line_count = 0

    dup_pmid = dict()
    while True:
        line = input_file.readline()
        if not line:
            # Reached the end of the input file
            break

        try:
            A = json.loads(line.strip()[:-1])
        except ValueError:
            continue

        line_count = line_count + 1
        if line_count % 10 is 0:
            sys.stdout.write(str(line_count) + '\r')
            sys.stdout.flush()

        pmid = A['pmid'].strip()
        if pmid in dup_pmid:
            continue
        else:
            dup_pmid[pmid] = 1

        doc = A['title'].strip() + ' ' + A['abstractText'].strip()
        mesh = A['meshMajor']
        year = int(re.match(r'\d+', A['year']).group())

        mesh_ids = list()
        for mesh_name in mesh:
            mesh_name = mesh_name.replace(' ', '_')
            if mesh_name.strip() in name_id_description_dict:
                mesh_ids.append(mesh_name.strip().replace(' ', '_'))
                # mesh_ids.append(name_id_description_dict[mesh_name.strip()][0].strip())

        if len(mesh_ids) == 0:
            continue

        if year < split_year:
            traindata.write('%s:::%s\n' %
                            (pmid.encode('utf-8'),
                             doc.encode('utf-8')))
            traindata.flush()
            trainlabel.write('%s:::%s\n' %
                             (pmid.encode('utf-8'),
                              (' '.join(mesh_ids))))
            trainlabel.flush()
        elif year >= split_year:
            testdata.write('%s:::%s\n' %
                           (pmid.encode('utf-8'),
                            doc.encode('utf-8')))
            testdata.flush()
            testlabel.write('%s:::%s\n' % (pmid, (' '.join(mesh_ids))))
            testlabel.flush()

if __name__ == '__main__':
    def split_year_constraint(year):
        year = int(year)
        if year < 2000 or year > 2014:
            raise argparse.ArgumentTypeError(
                "split data by year between 2000 and 2014")
        return year

    parser = argparse.ArgumentParser(
        description='Generate the BioASQ dataset used in the experiment.')
    parser.add_argument('-i', dest='input_file', required=True, metavar="FILE",
                        type=argparse.FileType('r'),
                        help="MeSH descriptors in XML format")
    parser.add_argument(
        '-m', '--mesh_info', dest='name_id_desc_mapping_file', required=True,
        metavar="FILE", type=argparse.FileType('r'),
        help="MeSH 2015 short")
    parser.add_argument(
        '--traindata_filepath', dest='traindata', required=True,
        metavar="FILE", type=argparse.FileType('w'),
        help="Training data filepath")
    parser.add_argument(
        '--trainlabel_filepath', dest='trainlabel', required=True,
        metavar="FILE", type=argparse.FileType('w'),
        help="Training label filepath")
    parser.add_argument(
        '--testdata_filepath', dest='testdata', required=True, metavar="FILE",
        type=argparse.FileType('w'),
        help="Testdata filepath")
    parser.add_argument(
        '--testlabel_filepath', dest='testlabel', required=True,
        metavar="FILE", type=argparse.FileType('w'),
        help="Testlabel filepath")
    parser.add_argument(
        '--split_year', dest='split_year', required=True,
        type=split_year_constraint, help="split data by this year")
    args = parser.parse_args()

    input_file = args.input_file
    name_id_desc_mapping_file = args.name_id_desc_mapping_file
    traindata = args.traindata
    trainlabel = args.trainlabel
    testdata = args.testdata
    testlabel = args.testlabel
    split_year = args.split_year

    process(input_file,
            name_id_desc_mapping_file,
            traindata,
            trainlabel,
            testdata,
            testlabel,
            split_year)
