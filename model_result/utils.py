import json
import os
import re
import sentencepiece as spm
from collections import Counter

def merge_json_files(input_folder, output_file):
    all_data = []
    
    # 모든 JSON 파일 처리
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 'data' 키 안의 리스트 추출
                if 'data' in data:
                    all_data.extend(data['data'])
                    print(f"  → {len(data['data'])}개 문장 쌍 추가됨")
    
    # 합친 데이터 저장
    merged_data = {"data": all_data}
    
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"총 {len(all_data)}개 문장 쌍이 합쳐졌습니다!")
    return len(all_data)



def is_korean(text):
    return bool(re.search(r'[가-힣]', text))

def is_english(text):
    return bool(re.search(r'[a-zA-Z]', text))

def has_spacing_issue(text):
    if re.search(r'\s{5,}', text):
        return True
    if len(text.split()) <= 1 and len(text) > 20:
        return True
    return False

def contains_bad_tags(text):
    tag_patterns = [
        r'<[^>]+>',
        r'&[a-z]+;',
        r'\[[^\]]{10,}\]',
        r'\{[^\}]{10,}\}',
        r'\([^)]{30,}\)',
    ]
    return any(re.search(p, text) for p in tag_patterns)

def clean_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    seen_pairs = set()
    min_len = 3
    MAX_LENGTH = 300  # 완화된 최대 길이

    drop_counter = Counter()

    for item in data['data']:
        korean = item['source_sentence'].strip()
        english = item['target_sentence'].strip()

        pair = (korean, english)
        if pair in seen_pairs:
            drop_counter['중복'] += 1
            continue

        if len(korean) < min_len or len(english) < min_len:
            drop_counter['짧은 문장'] += 1
            continue

        if len(korean) > MAX_LENGTH or len(english) > MAX_LENGTH:
            drop_counter['긴 문장'] += 1
            continue

        if not (is_korean(korean) or is_english(korean)):
            drop_counter['한국어/영어 없는 문장 (소스)'] += 1
            continue

        if not (is_korean(english) or is_english(english)):
            drop_counter['한국어/영어 없는 문장 (타겟)'] += 1
            continue

        if has_spacing_issue(korean) or has_spacing_issue(english):
            drop_counter['띄어쓰기 문제'] += 1
            continue

        if contains_bad_tags(korean) or contains_bad_tags(english):
            drop_counter['태그 문제'] += 1
            continue

        cleaned_data.append({
            'source_sentence': korean,
            'target_sentence': english,
            'domain': item.get('domain', 'unknown')
        })
        seen_pairs.add(pair)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'data': cleaned_data}, f, ensure_ascii=False, indent=2)

    total = len(data['data'])
    kept = len(cleaned_data)
    dropped = total - kept

    print(f"정제 전: {total}개 → 정제 후: {kept}개 (드랍 {dropped}개)")
    print("드랍 사유별 통계:")
    for reason, count in drop_counter.most_common():
        print(f"  {reason}: {count}개")

# opennmt format으로 변경
def convert_to_opennmt_format(json_file, ko_file, en_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(ko_file, 'w', encoding='utf-8') as ko_f, \
         open(en_file, 'w', encoding='utf-8') as en_f:
        
        for item in data['data']:
            # AI Hub 구조에서 문장 추출
            korean = item['source_sentence']
            english = item['target_sentence']
            
            ko_f.write(korean + '\n')
            en_f.write(english + '\n')
    
    print(f"{len(data['data'])}개 문장 쌍 변환 완료!")
    
# 한국어와 영어 텍스트 합치기 (토크나이저 학습용)
def prepare_tokenizer_data():
    with open('tokenizer_corpus.txt', 'w', encoding='utf-8') as f:
        # 한국어 데이터 추가
        with open('train.ko', 'r', encoding='utf-8') as ko_f:
            for line in ko_f:
                f.write(line)
        
        # 영어 데이터 추가
        with open('train.en', 'r', encoding='utf-8') as en_f:
            for line in en_f:
                f.write(line)
                


# SentencePiece로 토크나이징 (OpenNMT용)
def tokenize_data():
    sp = spm.SentencePieceProcessor()
    sp.load('sentencepiece.model')
    
    # Training 데이터 토크나이징
    tokenize_file(sp, 'train.ko', 'train.tok.ko')
    tokenize_file(sp, 'train.en', 'train.tok.en')
    
    # Validation 데이터 토크나이징
    tokenize_file(sp, 'valid.ko', 'valid.tok.ko')
    tokenize_file(sp, 'valid.en', 'valid.tok.en')



def tokenize_file(sp, input_file, output_file):
    """개별 파일 토크나이징"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            tokens = sp.encode_as_pieces(line.strip())
            f_out.write(' '.join(tokens) + '\n')



def merge_data_json(file1, file2, output_file):
    """
    {"data": [...]} 형태의 JSON 파일들을 합치는 함수
    """
    # 첫 번째 파일 읽기
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # 두 번째 파일 읽기
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # "data" 키 안의 리스트들 합치기
    merged_data = {
        "data": data1["data"] + data2["data"]
    }
    
    # 합쳐진 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 파일 합치기 완료!")
    print(f"📊 파일1 데이터: {len(data1['data'])}개")
    print(f"📊 파일2 데이터: {len(data2['data'])}개") 
    print(f"📊 합쳐진 데이터: {len(merged_data['data'])}개")
    print(f"💾 저장된 파일: {output_file}")
    
    return merged_data

