from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sentencepiece as spm
from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser
from contextlib import asynccontextmanager
from argparse import Namespace
import logging
from typing import Optional
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 안전한 글로벌 클래스 추가
torch.serialization.add_safe_globals([Namespace])

class TranslationRequest(BaseModel):
    text: str
    beam_size: Optional[int] = 5
    max_length: Optional[int] = 100
    min_length: Optional[int] = 0
    length_penalty: Optional[str] = 'none'  # 'none', 'wu', 'avg'
    coverage_penalty: Optional[str] = 'none'  # 'none', 'wu', 'summary'
    repetition_penalty: Optional[float] = 1.0
    no_repeat_ngram_size: Optional[int] = 0

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    beam_size: int
    max_length: int

translator = None
tokenizer = None

def load_tokenizer():
    global tokenizer
    try:
        tokenizer = spm.SentencePieceProcessor()
        model_path = "spm_model/spm.model"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
            
        tokenizer.load(model_path)
        logger.info("SentencePiece tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

def load_model():
    global translator
    try:
        model_path = "models/model_step_19000.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        opt = Namespace(
            models=[model_path],
            src="dummy",
            gpu=0 if torch.cuda.is_available() else -1,
            batch_size=1,
            beam_size=5,
            n_best=1,
            replace_unk=True,
            verbose=False,
            data_type='text',
            min_length=0,
            max_length=100,
            length_penalty='none',
            coverage_penalty='none',
            report_score=False,
            output="temp_output.txt",
            world_size=1,
            gpu_ranks=[0],
            precision="fp32",
            alpha=0.0,
            beta=-0.0
        )

        translator = build_translator(opt, report_score=False)
        logger.info("Translation model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_tokenizer()
        load_model()
        logger.info("Application startup completed")
        yield
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        logger.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    if translator is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 5000:  # 텍스트 길이 제한
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
    
    try:
        # 토크나이제이션
        tokens = tokenizer.encode(request.text, out_type=str)
        sentence = " ".join(tokens)
        
        # 번역 옵션 동적 설정
        translator.beam_size = request.beam_size
        translator.max_length = request.max_length
        translator.min_length = request.min_length
        
        # 번역 실행
        _, translations = translator.translate(
            src=[sentence],
            batch_size=1,
            attn_debug=False
        )
        
        if not translations or not translations[0]:
            raise Exception("Translation failed - no output generated")
        
        # 결과 디코딩
        raw_output = translations[0][0]
        result = tokenizer.decode(raw_output.split())
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=result,
            beam_size=request.beam_size,
            max_length=request.max_length
        )
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)