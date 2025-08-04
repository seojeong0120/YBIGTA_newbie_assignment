# PROMPTING.md

## 1. Accuracy 비교 (0-shot, 3-shot, 5-shot)

| Prompting 기법       | 0-shot | 3-shot | 5-shot |
|----------------------|--------|--------|--------|
| Direct Prompting     | 0.20   | 0.22   | 0.16   |
| Chain-of-Thought     | 0.80   | 0.70   | 0.66   |
| My Prompting         | 0.88   | 0.74   | 0.76   |

- **데이터 출처**  
  - `direct_prompting_0.txt`, `direct_prompting_3.txt`, `direct_prompting_5.txt`  
  - `CoT_prompting_0.txt`, `CoT_prompting_3.txt`, `CoT_prompting_5.txt`  
  - `My_prompting_0.txt`, `My_prompting_3.txt`, `My_prompting_5.txt`

---

## 2. 왜 CoT Prompting이 Direct Prompting보다 좋은가?

Chain-of-Thought (CoT) prompting은 문제를 한 번에 풀도록 요구하지 않고, **중간 사고 과정을 명시**하게 유도하는 프롬프트 방식이다. 이러한 구조 덕분에 Direct Prompting보다 다음과 같은 면에서 우수한 성능을 보인다.:

- **복잡한 계산 또는 논리 문제에서 성능 향상**  
  LLM이 한 번에 정답을 도출하기 어려운 문제에서도, 중간 단계를 거치며 더 안정적으로 정답에 도달할 수 있다.

- **추론 과정 자체가 문제 해결 힌트로 작용**  
  모델은 CoT 구조를 통해 자체적으로 오답을 검토하거나 오류를 줄일 기회를 가진다.

- **Human-like reasoning 유도**  
  인간이 문제를 푸는 방식과 유사한 형태로 모델의 사고 흐름을 유도하여 더 자연스럽고 일관된 답변을 생성하게 된다.

---

## 3. 내가 만든 프롬프트가 CoT보다 더 나은 이유

내가 구현한 `construct_my_prompt()` 함수는 다음과 같은 점에서 CoT보다 성능이 우수할 수 있었다:

### 1. **간결하고 안정된 지시문 구조**
- `"You are a brilliant math tutor..."` 문장으로 LLM의 역할을 명확히 설정했다.
- `"Step 1, Step 2, ..."`, `"Answer: ####"` 형식을 강제하여 **일관된 추론 패턴**을 학습하게 유도했다다.

### 2. **숫자 중심의 간결한 reasoning 유도**
- `"Keep reasoning short and numerical when possible."`이라는 명시적 지시를 통해,
  복잡하지 않은 수학 문제에 **불필요한 문장 생성을 억제**하고, 계산에 집중하도록 했다.

### 3. **예시 다양성과 구조 정규화**
- 덧셈, 곱셈, 단가 계산, 거리·면적 문제 등 다양한 문제 유형을 예시로 포함시켜,
  모델이 일반화된 수학적 reasoning 구조를 익힐 수 있게 했다.
- 모든 예시가 동일한 포맷(Question → Step-by-step → Answer)으로 구성되어 있어,
  학습 가능한 일관된 구조로 인식되도록 하였다.

### 4. **명확한 문제 전환부 구성**
```python
prompt += (
    "Now, solve the following question carefully.\n"
    "Question:\n{question}\n"
    "Let's think step by step:\n"
    "Answer:"
)
