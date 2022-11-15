# -*- coding: utf-8 -*-

import wave
import numpy as np
import os
import sys
from compute_mfcc import FeatureExtractor


#
# メイン関数
#
if __name__ == "__main__":

    #
    # 設定ここから
    #

    # 各wavファイルのリストと特徴量の出力先
    train_wav_scp = 'data/train/wav.scp'
    train_out_dir = 'data/wav/train'
    dev_wav_scp = 'data/dev/wav.scp'
    dev_out_dir = 'data/wav/dev'
    test_wav_scp = 'data/eval/wav.scp'
    test_out_dir = 'data/wav/eval'

    # wavファイルリストと出力先をリストにする
    wav_scp_list = [train_wav_scp,
                    dev_wav_scp,
                    test_wav_scp]
    out_dir_list = [train_out_dir,
                    dev_out_dir,
                    test_out_dir]

    FE = FeatureExtractor(
        sample_frequency=16000,
        frame_length=1024,
        frame_shift=256,
        num_mel_bins=80,
        num_ceps=13,
        lifter_coef=22,
        low_frequency=20,
        high_frequency=8000,
        dither=1.0
    )

    # 各セットについて処理を実行する
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print('Input wav_scp: %s' % (wav_scp))
        print('Output directory: %s' % (out_dir))

        # 特徴量ファイルのパス，フレーム数，
        # 次元数を記したリスト
        feat_scp = os.path.join(out_dir, 'feats.scp')

        # 出力ディレクトリが存在しない場合は作成する
        os.makedirs(out_dir, exist_ok=True)

        # wavリストを読み込みモード、
        # 特徴量リストを書き込みモードで開く
        old_speaker = "jvs000"
        with open(wav_scp, mode='r') as file_wav, \
                open(feat_scp, mode='w') as file_feat:
            # wavリストを1行ずつ読み込む
            for line in file_wav:
                # 各行には，発話IDとwavファイルのパスが
                # スペース区切りで記載されているので，
                # split関数を使ってスペース区切りの行を
                # リスト型の変数に変換する
                parts = line.split()
                # 0番目が発話ID
                utterance_id = parts[0]
                # 1番目がwavファイルのパス
                wav_path = parts[1]

                # 今何を処理しているのかを明確にしている
                if (old_speaker != utterance_id.split(sep='_')[0]):
                    old_speaker = utterance_id.split(sep='_')[0]
                    print(f"      {old_speaker}")

                # wavファイルを読み込み，特徴量を計算する
                with wave.open(wav_path) as wav:
                    # wavファイルが1チャネル(モノラル) データであることをチェック
                    if wav.getnchannels() != 1:
                        sys.stderr.write('This program supports monaural wav file only.\n')
                        exit(1)

                    # wavデータのサンプル数
                    num_samples = wav.getnframes()

                    # wavデータを読み込む
                    waveform = wav.readframes(num_samples)

                    # 読み込んだデータはバイナリ値
                    # (16bit integer)なので，数値(整数)に変換する
                    waveform = np.frombuffer(waveform, dtype=np.int16)

                    # int16 to float16
                    wav_float = waveform.astype(np.float16) / np.power(2, 15)

                # 特徴量ファイルの名前(splitextで拡張子を取り除いている)
                out_file = utterance_id
                out_file = os.path.join(os.path.abspath(out_dir),
                                        out_file + '.bin')

                # データをファイルに出力
                wav_float.tofile(out_file)
                # 発話ID，特徴量ファイルのパス，フレーム数，
                # 次元数を特徴量リストに書き込む
                file_feat.write("%s %s %d %d\n" %
                                (utterance_id, out_file, num_samples, 1))
