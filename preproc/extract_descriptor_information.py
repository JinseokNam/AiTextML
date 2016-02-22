import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(
    description=("Extract Descriptor names, IDs, and"
                 "descriptions from MeSH 2015 in XML format."
                 "MeSH 2015 can be downloaded at"
                 "http://www.nlm.nih.gov/mesh/filelist.html"))
parser.add_argument(
    '-i', dest='input_file', required=True, metavar="FILE",
    type=argparse.FileType('r'),
    help="MeSH descriptors in XML format")
parser.add_argument(
    '-o', dest='output_file', required=True, metavar="FILE",
    type=argparse.FileType('w'),
    help="Output which only contains names, IDs, and descriptions")
args = parser.parse_args()

fin = args.input_file
fout = args.output_file

tree = ET.parse(fin)
root = tree.getroot()

for descriptor in root.findall('DescriptorRecord'):
    name = descriptor.find('DescriptorName').find('String').text
    name = name.replace(' ', '_')
    # ui = descriptor.find('DescriptorUI').text
    scopenote = descriptor.find('ConceptList').find(
        'Concept').find('ScopeNote')
    if scopenote is not None:
        description = scopenote.text.strip()
    else:
        description = ''
    """
    fout.write('%s:::%s:::%s\n' %
               (name.encode('utf-8'),
                ui,
                description.encode('utf-8')))
    """
    fout.write('%s:::%s\n' %
               (name.encode('utf-8'),
                description.encode('utf-8')))
    fout.flush()

fin.close()
fout.close()
