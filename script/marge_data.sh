#!/bin/bash

# Project Path: デフォルトでは、実行したディレクトリ
PROJECT_PATH=`pwd`

# ACTOR path
DATA_PATH="data"
ACTOR_PATH="actor"
TRAIN_PATH="train"
DEV_PATH="dev"
EVAL_PATH="eval"

# export path
EXP_TRAIN_PATH="${PROJECT_PATH}/${DATA_PATH}/${TRAIN_PATH}"
EXP_DEV_PATH="${PROJECT_PATH}/${DATA_PATH}/${DEV_PATH}"
EXP_EVAL_PATH="${PROJECT_PATH}/${DATA_PATH}/${EVAL_PATH}"

# actor list
if [ $# -eq 0 ];then
    ACTOR_LIST="marge_actor_list_v1.csv"
elif [ $# -eq 1 ];then
    ACTOR_LIST=$1
fi

# INPUT: $1, OUTPUT: $2, ACTOR: $3, LABEL: $4
function merge_data_label() {
    cat $1/text >> $2/text
    cat $1/spk2utt >> $2/spk2utt
    cat $1/utt2spk >> $2/utt2spk
    cat $1/wav.scp >> $2/wav.scp
    cat $1/segments >> $2/segments

    # create label
    SED_OPTION="s/$3$/$4/g"
    sed "${SED_OPTION}" $1/utt2spk >> $2/label
    unset SED_OPTION
}

# ディレクトリを作成
mkdir -p ${EXP_TRAIN_PATH} ${EXP_DEV_PATH} ${EXP_EVAL_PATH}

# 既にディレクトリ内にあるデータを削除
rm -f ${EXP_TRAIN_PATH}/* ${EXP_DEV_PATH}/* ${EXP_EVAL_PATH}/*

for line in $(cat ${DATA_PATH}/${ACTOR_LIST});do
    ACTOR=$(cut -d',' -f1 <(echo ${line}))
    LABEL=$(cut -d',' -f2 <(echo ${line}))
    echo -n ${ACTOR}

    merge_data_label "${PROJECT_PATH}/${DATA_PATH}/${ACTOR_PATH}/${ACTOR}/${TRAIN_PATH}" ${EXP_TRAIN_PATH} ${ACTOR} ${LABEL}
    merge_data_label "${PROJECT_PATH}/${DATA_PATH}/${ACTOR_PATH}/${ACTOR}/${DEV_PATH}" ${EXP_DEV_PATH} ${ACTOR} ${LABEL}
    merge_data_label "${PROJECT_PATH}/${DATA_PATH}/${ACTOR_PATH}/${ACTOR}/${EVAL_PATH}" ${EXP_EVAL_PATH} ${ACTOR} ${LABEL}

    echo " DONE!!!"
done

# 生成したTMPを削除
if [ -f text_tmp ];then rm text_tmp; fi

# 変数を削除
unset PROJECT_PATH ACTOR DATA_PATH ACTOR_PATH TRAIN_PATH DEV_PATH EVAL_PATH EXP_TRAIN_PATH EXP_DEV_PATH EXP_EVAL_PATH merge_data
