import sentencepiece as smp
import torch
from onmt.translate.translator import build_translator
import argparse
from onmt.opts import translate_opts


class KoreanEnglishTranslator:
    def __init__(self, model_path, tokenizer_path):
        self.sp = smp.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        self.opt = self._parse_args(model_path, tokenizer_path)
        self.translator = build_translator(self.opt, report_score=False)

    def _parse_args(self, model_path, tokenizer_path):
        parser = argparse.ArgumentParser()
        translate_opts(parser)

        args = parser.parse_args([
            '--model', model_path,
            '--beam_size', '5',
            '--batch_size', '1',
            '--replace_unk',
            '--src', 'dummy.txt',
            '--src_subword_model', tokenizer_path,
            '--tgt_subword_model', tokenizer_path,
        ] + (['--gpu', '0'] if torch.cuda.is_available() else []))

        return args

    def translate(self, korean_text):
        korean_text = korean_text.strip()
        if not korean_text:
            return ""

        tokens = self.sp.encode(korean_text, out_type=int)
        print(f"tokens: {tokens}")

        src_tensor = torch.LongTensor(tokens).unsqueeze(0).unsqueeze(-1)  # (1, seq_len)
        print(f"src_tensor.shape: {src_tensor.shape}")

        src_len = torch.LongTensor([len(tokens)])
        print(f"src_len.shape: {src_len.shape}, src_len: {src_len}")

        if torch.cuda.is_available():
            src_tensor = src_tensor.cuda()
            src_len = src_len.cuda()

        batch = {
            "src": src_tensor,
            "srclen": src_len
        }

        # ✅ 여기 테스트 코드 넣기
        try:
            embedding = self.translator.model.encoder.embeddings
            sample_input = src_tensor
            if torch.cuda.is_available():
                embedding = embedding.cuda()
            output = embedding(sample_input)
            print("✅ [테스트] embedding output shape:", output.shape)
        except Exception as e:
            print(f"🚨 [테스트 오류] embedding 오류: {e}")

        # 6. 번역 실행
        try:
            scores, predictions = self.translator.translate_batch(batch, attn_debug=False)
            print(f"scores: {scores}, predictions: {predictions}")

            # 7. 결과 디코딩
            if predictions and len(predictions[0]) > 0:
                pred_ids = predictions[0][0]
                # EOS, PAD, UNK, BOS 토큰 필터링 (일반적으로 0~3)
                filtered_ids = [t for t in pred_ids if t >= 4]
                english_text = self.sp.decode(filtered_ids)
                return english_text

        except Exception as e:
            print(f"Translation error: {e}")
            return ""

        return ""
