#!/bin/bash

SIZE=`du -sk $1| cut -f 1`

tr " " "\n" < $1 | pv -p -r -t -e -s ${SIZE}k -cN create_vocabulary.sh | sort | uniq -c | sort -bnr | awk -v OFS='\t' '{ print $2, $1 }' > $2
