# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    # Hugging Face의 "poem sentiment" 데이터셋 로드
    dataset = load_dataset("google-research-datasets/poem_sentiment")

    # train + validation + test 데이터셋에서 텍스트 수집
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            sentence = item["verse_text"]
            corpus.append(sentence)
    return corpus