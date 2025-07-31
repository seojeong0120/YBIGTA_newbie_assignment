import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        # padding token id -> 학습 중 제거에 사용
        self.pad_token_id = None # 이후 tokenizer에서 설정

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        # 토큰화된 corpus를 사용하여 학습
        for epoch in range(num_epochs):
            # cbow 방식일 경우 학습
            if self.method == "cbow":
                self._train_cbow(corpus, tokenizer, criterion, optimizer)
            # skipgram 방식일 경우 학습
            elif self.method == "skipgram":
                self._train_skipgram(corpus, tokenizer, criterion, optimizer)

    def _train_cbow(
        self,
        corpus, tokenizer, criterion, optimizer
    ) -> None:
        # 구현하세요!
        for sentence in corpus:
            # 문장을 토큰화
            tokens = tokenizer(sentence, add_special_tokens=False)

            # padding token 제거
            tokens = [token for token in tokens if token != self.pad_token_id]

            # 중심 단어 위치 기준 범위 설정
            for center_idx in range(self.window_size, len(tokens) - self.window_size):
                # 주변 단어
                context = (
                    tokens[center_idx - self.window_size:center_idx] +
                    tokens[center_idx + 1:center_idx + self.window_size + 1]
                )
                # 중심 단어
                target = tokens[center_idx]
                # context 또는 target에 padding에 있으면 skip
                if target == self.pad_token_id or any(token == self.pad_token_id for token in context):
                    continue
                # tesnor로 변환
                context_tensor = torch.tensor(context).to(self.embeddings.weight.device) # (2 * window_size,)
                target_tensor = torch.tensor([target]).to(self.embeddings.weight.device) # (1,)
                # context 임베딩 평균
                context_embedding = self.embeddings(context_tensor) # (2 * window_size, d_model)
                context_mean = context_embedding.mean(dim=0) # (d_model,)
                # 예측 확률 분포 생성
                logits = self.weight(context_mean) # (vocab_size,)
                # 손실 계산
                loss = criterion(logits.unsqueeze(0), target_tensor) # (1, vocab_size)
                # 기존 기울기 초기화
                optimizer.zero_grad()
                # 역전파 수행하여 그래디언트 계산
                loss.backward()
                # 파라미터 업데이트
                optimizer.step()

    def _train_skipgram(
        self,
        corpus: list[str],
        tokenizer,
        criterion,
        optimizer
    ) -> None:
        # 구현하세요!
        for sentence in corpus:
            # 문장을 토큰화
            tokens = tokenizer.encode(sentence, add_special_tokens=False)

            # padding token 제거
            tokens = [token for token in tokens if token != self.pad_token_id]

            # 중심 단어 위치 기준 범위 설정
            for center_idx in range(self.window_size, len(tokens) - self.window_size):
                # 중심 단어
                center = tokens[center_idx]
                # 주변 단어
                context = (
                    tokens[center_idx - self.window_size:center_idx] +
                    tokens[center_idx + 1:center_idx + self.window_size + 1]
                )
                # padding token 있는 경우 skip
                if center == self.pad_token_id or any(token == self.pad_token_id for token in context):
                    continue
                # tensor로 변환
                center_tensor = torch.tensor([center]).to(self.embeddings.weight.device) # (1,)
                # 중심 단어 임베딩
                center_embedding = self.embeddings(center_tensor).squeeze(0).detach() # (d_model,)
                for ctx in context:
                    # 정답 레이블인 context 단어 인덱스를 텐서로 변환
                    context_tensor = torch.tensor([ctx]).to(self.embeddings.weight.device) # (1,)
                    # 중심 단어의 임베딩을 이용해 예측 확률 분포 생성
                    logits = self.weight(center_embedding) # (vocab_size,)
                    # 손실 계산
                    loss = criterion(logits.unsqueeze(0), context_tensor)
                    # 기존 기울기 초기화
                    optimizer.zero_grad()
                    # 역전파 수행하여 그래디언트 계산
                    loss.backward()
                    # 파라미터 업데이트
                    optimizer.step()