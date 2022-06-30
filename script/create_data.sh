#!/bin/bash

# Project Path: デフォルトでは、実行したディレクトリ
PROJECT_PATH=`pwd`

# file path
WAV_JVS_PATH="downloads/jvs_ver1"
WAV_TRAIN_PATH="parallel100"
WAV_DEV_PATH="nonpara30"  # 個数が15個までなことに注意
WAV_EVAL_PATH="nonpara30"  # dev以外の残りの15個
WAV_DIR_NAME="wav24kHz16bit"
TRANSCRIPT_NAME="transcripts_utf8.txt"
SEGMENTS_MAME="lab/mon"

# export path
EXP_DATA_PATH="data"
EXP_ACTOR_PATH="actor"
EXP_TRAIN_PATH="train"
EXP_DEV_PATH="dev"
EXP_EVAL_PATH="eval"

for i in $(seq --format='%03.0f' 1 100);do
    ACTOR="jvs${i}"
    echo -n ${ACTOR}

    # TRAIN
    echo -n " TRAIN"
    EXP_TMP_PATH="${PROJECT_PATH}/${EXP_DATA_PATH}/${EXP_ACTOR_PATH}/${ACTOR}/${EXP_TRAIN_PATH}"
    WAV_TMP_PATH="${PROJECT_PATH}/${WAV_JVS_PATH}/${ACTOR}/${WAV_TRAIN_PATH}"
    TEXT_TARGET="${EXP_TMP_PATH}/text"
    SPKUTT_TARGET="${EXP_TMP_PATH}/spk2utt"
    UTTSPK_TARGET="${EXP_TMP_PATH}/utt2spk"
    WAV_TARGET="${EXP_TMP_PATH}/wav.scp"
    SEGMENTS_TARGET="${EXP_TMP_PATH}/segments"
    # ディレクトリがないなら作る
    mkdir -p ${EXP_TMP_PATH}

    WAV_FNUM=$(ls ${WAV_TMP_PATH}/${WAV_DIR_NAME} | wc -l)

    # ファイルの初期化
    if [ -f ${WAV_TARGET} ];then rm ${WAV_TARGET}; fi
    if [ -f ${SEGMENTS_TARGET} ];then rm ${SEGMENTS_TARGET}; fi
    if [ -f ${TEXT_TARGET} ];then rm ${TEXT_TARGET}; fi
    if [ -f ${UTTSPK_TARGET} ];then rm ${UTTSPK_TARGET}; fi
    for file in `echo ${WAV_TMP_PATH}/${WAV_DIR_NAME}/*`;do
        FNAME="$(basename -s '.wav' ${file})" # ファイル名
        UTT_ID="${ACTOR}_${FNAME}" # 発話ID
        # wav.scpを生成(追記)
        echo "${UTT_ID} ${file##$PROJECT_PATH/}" >> ${WAV_TARGET}
        # segmentsを生成
        STA_TIME=`head -n 1 "${WAV_TMP_PATH}/${SEGMENTS_MAME}/${FNAME}.lab" | cut -d' ' -f2`
        END_TIME=`tail -n 1 "${WAV_TMP_PATH}/${SEGMENTS_MAME}/${FNAME}.lab" | cut -d' ' -f1`
        echo "${UTT_ID} ${UTT_ID} ${STA_TIME} ${END_TIME}" >> ${SEGMENTS_TARGET}
        # textを生成
        paste -d'_' <(echo ${ACTOR}) <(grep ${FNAME} ${WAV_TMP_PATH}/${TRANSCRIPT_NAME}) |\
            sed 's/:/ /g' >> ${TEXT_TARGET}
        # utt2spkを生成
        echo "${UTT_ID} ${ACTOR}" >> ${UTTSPK_TARGET}
    done
    # spk2uttを生成
    echo "${ACTOR} $(cut -d' ' -f1 ${UTTSPK_TARGET} | tr '\n' ' ' | sed 's/ $//g')" > ${SPKUTT_TARGET}


    # DEV
    echo -n " DEV"
    EXP_TMP_PATH="${PROJECT_PATH}/${EXP_DATA_PATH}/${EXP_ACTOR_PATH}/${ACTOR}/${EXP_DEV_PATH}"
    WAV_TMP_PATH="${PROJECT_PATH}/${WAV_JVS_PATH}/${ACTOR}/${WAV_DEV_PATH}"
    TEXT_TARGET="${EXP_TMP_PATH}/text"
    SPKUTT_TARGET="${EXP_TMP_PATH}/spk2utt"
    UTTSPK_TARGET="${EXP_TMP_PATH}/utt2spk"
    WAV_TARGET="${EXP_TMP_PATH}/wav.scp"
    SEGMENTS_TARGET="${EXP_TMP_PATH}/segments"
    # ディレクトリがないなら作る
    mkdir -p ${EXP_TMP_PATH}

    WAV_FNUM=$(ls ${WAV_TMP_PATH}/${WAV_DIR_NAME} | wc -l)

    EXP_COUNTER=0  # カウンタ
    if [ -f ${WAV_TARGET} ];then rm ${WAV_TARGET}; fi
    if [ -f ${SEGMENTS_TARGET} ];then rm ${SEGMENTS_TARGET}; fi
    if [ -f ${TEXT_TARGET} ];then rm ${TEXT_TARGET}; fi
    if [ -f ${UTTSPK_TARGET} ];then rm ${UTTSPK_TARGET}; fi
    for file in `echo ${WAV_TMP_PATH}/${WAV_DIR_NAME}/*`;do
        FNAME="$(basename -s '.wav' ${file})" # ファイル名
        UTT_ID="${ACTOR}_${FNAME}" # 発話ID
        # wav.scpを生成(追記)
        if (( EXP_COUNTER++ < ( WAV_FNUM / 2 ) ));then
            echo "${UTT_ID} ${file##$PROJECT_PATH/}" >> ${WAV_TARGET}
            # segmentsを生成
            STA_TIME=`head -n 1 "${WAV_TMP_PATH}/${SEGMENTS_MAME}/${FNAME}.lab" | cut -d' ' -f2`
            END_TIME=`tail -n 1 "${WAV_TMP_PATH}/${SEGMENTS_MAME}/${FNAME}.lab" | cut -d' ' -f1`
            echo "${UTT_ID} ${UTT_ID} ${STA_TIME} ${END_TIME}" >> ${SEGMENTS_TARGET}
            # textを生成
            paste -d'_' <(echo ${ACTOR}) <(grep ${FNAME} ${WAV_TMP_PATH}/${TRANSCRIPT_NAME}) |\
                sed 's/:/ /g' >> ${TEXT_TARGET}
            # utt2spkを生成
            echo "${UTT_ID} ${ACTOR}" >> ${UTTSPK_TARGET}
        else
            break
        fi
    done
    # spk2uttを生成
    echo "${ACTOR} $(cut -d' ' -f1 ${UTTSPK_TARGET} | tr '\n' ' ' | sed 's/ $//g')" > ${SPKUTT_TARGET}


    # EVAL
    echo -n " EVAL"
    EXP_TMP_PATH="${PROJECT_PATH}/${EXP_DATA_PATH}/${EXP_ACTOR_PATH}/${ACTOR}/${EXP_EVAL_PATH}"
    WAV_TMP_PATH="${PROJECT_PATH}/${WAV_JVS_PATH}/${ACTOR}/${WAV_EVAL_PATH}"
    TEXT_TARGET="${EXP_TMP_PATH}/text"
    SPKUTT_TARGET="${EXP_TMP_PATH}/spk2utt"
    UTTSPK_TARGET="${EXP_TMP_PATH}/utt2spk"
    WAV_TARGET="${EXP_TMP_PATH}/wav.scp"
    SEGMENTS_TARGET="${EXP_TMP_PATH}/segments"
    # ディレクトリがないなら作る
    mkdir -p ${EXP_TMP_PATH}

    WAV_FNUM=$(ls ${WAV_TMP_PATH}/${WAV_DIR_NAME} | wc -l)

    EXP_COUNTER=0  # カウンタ
    if [ -f ${WAV_TARGET} ];then rm ${WAV_TARGET}; fi
    if [ -f ${SEGMENTS_TARGET} ];then rm ${SEGMENTS_TARGET}; fi
    if [ -f ${TEXT_TARGET} ];then rm ${TEXT_TARGET}; fi
    if [ -f ${UTTSPK_TARGET} ];then rm ${UTTSPK_TARGET}; fi
    for file in `echo ${WAV_TMP_PATH}/${WAV_DIR_NAME}/*`;do
        FNAME="$(basename -s '.wav' ${file})" # ファイル名
        UTT_ID="${ACTOR}_${FNAME}" # 発話ID
        # wav.scpを生成(追記)
        if (( EXP_COUNTER++ >= ( WAV_FNUM / 2 ) ));then
            echo "${UTT_ID} ${file##$PROJECT_PATH/}" >> ${WAV_TARGET}
            # segmentsを生成
            STA_TIME=`head -n 1 "${WAV_TMP_PATH}/${SEGMENTS_MAME}/${FNAME}.lab" | cut -d' ' -f2`
            END_TIME=`tail -n 1 "${WAV_TMP_PATH}/${SEGMENTS_MAME}/${FNAME}.lab" | cut -d' ' -f1`
            echo "${UTT_ID} ${UTT_ID} ${STA_TIME} ${END_TIME}" >> ${SEGMENTS_TARGET}
            # textを生成
            paste -d'_' <(echo ${ACTOR}) <(grep ${FNAME} ${WAV_TMP_PATH}/${TRANSCRIPT_NAME}) |\
                sed 's/:/ /g' >> ${TEXT_TARGET}
            # utt2spkを生成
            echo "${UTT_ID} ${ACTOR}" >> ${UTTSPK_TARGET}
        fi
    done
    # spk2uttを生成
    echo "${ACTOR} $(cut -d' ' -f1 ${UTTSPK_TARGET} | tr '\n' ' ' | sed 's/ $//g')" > ${SPKUTT_TARGET}

    echo " DONE!!!"
done

# 生成したTMPを削除
if [ -f text_tmp ];then rm text_tmp; fi

# 変数を削除
unset ACTOR
unset WAV_JVS_PATH
unset WAV_TRAIN_PATH
unset WAV_DEV_PATH
unset WAV_EVAL_PATH
unset EXP_DATA_PATH
unset EXP_ACTOR_PATH
unset EXP_TRAIN_PATH
unset EXP_DEV_PATH
unset EXP_EVAL_PATH
unset EXP_TMP_PATH
unset WAV_TMP_PATH
unset EXP_COUNTER
unset TEXT_TARGET
unset SPKUTT_TARGET
unset UTTSPK_TARGET
unset WAV_TARGET
unset SEGMENTS_TARGET
