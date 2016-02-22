#!/bin/bash

BIN_PATH=../bin

ORIGINAL_DATA=$1
NAME_ID_MAPPING=$2
OUTPUT_DIR=$3

if [[ ! -e $OUTPUT_DIR ]]; then
  mkdir $OUTPUT_DIR
elif [[ ! -d $OUTPUT_DIR ]]; then
  echo "$OUTPUT_DIR already exists but is not a directory" 1>&2
fi

NAME_DESC=${OUTPUT_DIR}/MeSH_name_description.txt
TRD=${OUTPUT_DIR}/trd.txt
TRL=${OUTPUT_DIR}/trl.txt
VAD=${OUTPUT_DIR}/vad.txt
VAL=${OUTPUT_DIR}/val.txt
TSD=${OUTPUT_DIR}/tsd.txt
TSL=${OUTPUT_DIR}/tsl.txt
SPLIT_YEAR=2005

TRD_TOK=${OUTPUT_DIR}/trd.tokenized.txt
TSD_TOK=${OUTPUT_DIR}/tsd.tokenized.txt
TRD_TOK_SHUF=${OUTPUT_DIR}/trd.tokenized.shuf.txt
VAD_TOK_SHUF=${OUTPUT_DIR}/vad.tokenized.shuf.txt
TRL_SHUF=${OUTPUT_DIR}/trl.shuf.txt
VAL_SHUF=${OUTPUT_DIR}/val.shuf.txt
NAME_DESC_TOK=${OUTPUT_DIR}/MeSH_name_description.tokenized.txt

WORD_VOCAB=${OUTPUT_DIR}/word_vocabulary.txt
WORD_MIN_CNT=5
LABEL_MIN_CNT=1
SEEN_LABEL_VOCAB=${OUTPUT_DIR}/seen_label_vocabulary.txt
TOTAL_LABEL_VOCAB=${OUTPUT_DIR}/total_label_vocabulary.txt

HDF5FILE_PATH=${OUTPUT_DIR}/dataset.h5

python extract_descriptor_information.py -i ${NAME_ID_MAPPING} -o ${NAME_DESC}
python create_BioASQ_dataset.py -i ${ORIGINAL_DATA} -m ${NAME_DESC} --traindata_filepath ${TRD} --trainlabel_filepath ${TRL} --testdata_filepath ${TSD} --testlabel_filepath ${TSL} --split_year ${SPLIT_YEAR}

TRD_ID_TMP="$(mktemp)"
TSD_ID_TMP="$(mktemp)"
NAME_DESC_TMP="$(mktemp)"

# preprocess text
awk -F":::" '{print $1}' ${TRD} | pv -p -r -t -e -cN creating-temporary-file > ${TRD_ID_TMP}
awk -F":::" '{print $2}' ${TRD} | perl normalize-punctuation.perl -l en | perl tokenizer.perl -l en | awk 'BEGIN { OFS=":::"} FNR==NR { a[(FNR"")] = $0; next } { print a[(FNR"")], $0 }' ${TRD_ID_TMP} - > ${TRD_TOK}

awk -F":::" '{print $1}' ${TSD} | pv -p -r -t -e -cN creating-temporary-file > ${TSD_ID_TMP}
awk -F":::" '{print $2}' ${TSD} | perl normalize-punctuation.perl -l en | perl tokenizer.perl -l en | awk 'BEGIN { OFS=":::"} FNR==NR { a[(FNR"")] = $0; next } { print a[(FNR"")], $0 }' ${TSD_ID_TMP} - > ${TSD_TOK}

awk -F":::" '{print $1}' ${NAME_DESC} | pv -p -r -t -e -cN creating-temporary-file > ${NAME_DESC_TMP}
awk -F":::" '{print $2}' ${NAME_DESC} | perl normalize-punctuation.perl -l en | perl tokenizer.perl -l en | awk 'BEGIN { OFS=":::"} FNR==NR { a[(FNR"")] = $0; next } { print a[(FNR"")], $0 }' ${NAME_DESC_TMP} - > ${NAME_DESC_TOK}

# split the original train data into train and validation data
MERGED_TMP="$(mktemp)"
echo $MERGED_TMP
VAD_SIZE=100000
TR_NUM_LINE="$(wc -l ${TRL} | cut -d' ' -f 1)"
paste ${TRL} ${TRD_TOK} | shuf > ${MERGED_TMP}

cut -f 1 -d$'\t' ${MERGED_TMP} | head -n ${VAD_SIZE} > ${VAL_SHUF}
cut -f 2 -d$'\t' ${MERGED_TMP} | head -n ${VAD_SIZE} > ${VAD_TOK_SHUF}
cut -f 1 -d$'\t' ${MERGED_TMP} | tail -n $((TR_NUM_LINE-VAD_SIZE)) > ${TRL_SHUF}
cut -f 2 -d$'\t' ${MERGED_TMP} | tail -n $((TR_NUM_LINE-VAD_SIZE)) > ${TRD_TOK_SHUF}

# create word vocabulary from the train data
echo "  Now building word vocabulary"
awk -F":::" '{print $2}' ${TRD_TOK_SHUF} | tr " " "\n" | sort | uniq -c | sort -bnr | awk -v OFS='\t' '{ print $2, $1 }' > ${WORD_VOCAB}
echo "  word vocabulary has been stored at ${WORD_VOCAB}"
echo ""

# create label vocabulary from the train label
echo "  Now building seen label vocabulary"
echo ""
awk -F":::" '{print $2}' ${TRL_SHUF} | tr " " "\n" | sort | uniq -c | sort -bnr | awk -v OFS='\t' '{ print $2, $1 }' > ${SEEN_LABEL_VOCAB}
echo "  senn label vocabulary has been stored at ${SEEN_LABEL_VOCAB}"
echo ""

echo "  Now building total label vocabulary"
echo ""
cat ${TRL_SHUF} ${VAL_SHUF} ${TSL} | awk -F":::" '{print $2}' | tr " " "\n" | sort | uniq -c | sort -bnr | awk -v OFS='\t' '{ print $2, $1 }' > ${TOTAL_LABEL_VOCAB}
echo "  total label vocabulary has been stored at ${TOTAL_LABEL_VOCAB}"
echo ""

${BIN_PATH}/hdf5_converter --traindata_path ${TRD_TOK_SHUF} --trainlabel_path ${TRL_SHUF} --validdata_path ${VAD_TOK_SHUF} --validlabel_path ${VAL_SHUF} --testdata_path ${TSD_TOK} --testlabel ${TSL} --label_description_path ${NAME_DESC_TOK} --word_vocabulary ${WORD_VOCAB} --word_min_count ${WORD_MIN_CNT} --seen_label_vocabulary ${SEEN_LABEL_VOCAB} --total_label_vocabulary ${TOTAL_LABEL_VOCAB} --label_min_count ${LABEL_MIN_CNT} --output_filepath ${HDF5FILE_PATH}
