import sentencepiece as smp
import torch
from onmt.translate.translator import build_translator
import argparse
from onmt.opts import translate_opts


class KoreanEnglishTranslator:
    def __init__(self, model_path, tokenizer_path):
        # SentencePiece 토크나이저 로드
        self.sp = smp.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # OpenNMT 모델 로드
        self.translator = self._load_translator(model_path)
    
    def _load_translator(self, model_path):
        parser = argparse.ArgumentParser()
        translate_opts(parser)
        
        args = parser.parse_args([
            '--model', model_path,
            '--beam_size', '5',
            '--batch_size', '1',
            '--replace_unk',
            '--src', 'dummy.txt'
        ] + (['--gpu', '0'] if torch.cuda.is_available() else []))
        
        translator = build_translator(args, report_score=False)
        return translator
    
    def translate(self, korean_text):
        import os
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 에러 정확히 확인

        korean_text = korean_text.strip()
        if not korean_text:
            return ""

        # SentencePiece 토크나이징
        tokens = self.sp.encode_as_pieces(korean_text)
        
        # 토큰 ID 추출 + 검사
        token_ids = []
        for token in tokens:
            token_id = self.sp.piece_to_id(token)
            if token_id < 0:
                print(f"[경고] vocab에 없는 token: {token}, ID: {token_id}")
                token_id = self.sp.unk_id()
            token_ids.append(token_id)

        print("👉 token_ids:", token_ids)

        # vocab size 확인 - 주석 해제 및 개선
        try:
            vocab_size = self.translator.model.encoder.embeddings.make_embedding.emb_luts[0].num_embeddings
        except AttributeError:
            try:
                vocab_size = self.translator.model.encoder.embeddings.emb_luts[0].num_embeddings
            except AttributeError:
                try:
                    vocab_size = self.translator.model.src_embeddings.num_embeddings
                except AttributeError:
                    vocab_size = self.sp.get_piece_size()
                    print(f"[경고] 모델 vocab size 자동 감지 실패, SentencePiece vocab size 사용: {vocab_size}")
        
        print(f"📊 모델 vocab size: {vocab_size}")
        print(f"📊 SentencePiece vocab size: {self.sp.get_piece_size()}")
        
        # vocab size 범위 체크
        validated_token_ids = []
        for idx in token_ids:
            if idx >= vocab_size:
                print(f"[경고] token_id {idx}는 vocab_size {vocab_size} 초과 -> UNK({self.sp.unk_id()})로 변경")
                validated_token_ids.append(self.sp.unk_id())
            else:
                validated_token_ids.append(idx)
        
        print("✅ 검증된 token_ids:", validated_token_ids)

        # 텐서 생성 - 차원 수정: [seq_len, batch_size] 형태로 변경
        src_tensor = torch.tensor(validated_token_ids, dtype=torch.long).unsqueeze(1)  # [seq_len, 1]
        src_len = torch.tensor([len(validated_token_ids)], dtype=torch.long)

        print("✅ 텐서 생성 완료:", src_tensor.shape, src_tensor.dtype)
        print("✅ 길이 텐서:", src_len.shape, src_len.dtype)

        # GPU 전송 전 shape/type 검사
        if torch.any(src_tensor < 0):
            raise ValueError("[에러] src_tensor에 음수 ID 포함")
        if torch.any(src_tensor >= vocab_size):
            raise ValueError(f"[에러] src_tensor에 vocab_size({vocab_size}) 초과 ID 포함")
        if src_tensor.dtype != torch.long:
            raise ValueError("[에러] src_tensor는 long 타입이어야 합니다")

        # GPU 이동
        if torch.cuda.is_available():
            src_tensor = src_tensor.cuda()
            src_len = src_len.cuda()

        batch = {
            "src": src_tensor,
            "srclen": src_len,
            "src_map": None
        }

        print("✅ batch 생성 완료:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch.items()})

        # 번역 실행
        try:
            scores, predictions = self.translator.translate_batch(batch, attn_debug=False)
        except Exception as e:
            print(f"[에러] translate_batch 실행 중 오류: {e}")
            print(f"src_tensor.shape: {src_tensor.shape}")
            print(f"src_len.shape: {src_len.shape}")
            print(f"src_tensor 값 범위: {src_tensor.min().item()} ~ {src_tensor.max().item()}")
            raise

        # 결과 처리
        if predictions and len(predictions) > 0:
            prediction_ids = predictions[0][0]  # 첫 문장의 첫 beam 결과

            # ID를 토큰으로 변환
            predicted_tokens = []
            for idx in prediction_ids:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()

                # 특수 토큰 제외
                if idx not in [self.sp.piece_to_id('<pad>'), 
                            self.sp.piece_to_id('</s>'), 
                            self.sp.piece_to_id('<s>'),
                            self.sp.unk_id()]:
                    token = self.sp.id_to_piece(idx)
                    predicted_tokens.append(token)

            # 디코딩 (token -> 문자열)
            english_text = self.sp.decode_pieces(predicted_tokens)
            return english_text

        return ""