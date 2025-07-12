# lib.py의 Matrix 클래스를 참조하지 않음
import sys


"""
TODO:
- fast_power 구현하기 
"""


def fast_power(base: int, exp: int, mod: int) -> int:
    """
    빠른 거듭제곱 알고리즘 구현
    분할 정복을 이용, 시간복잡도 고민!
    """
    # 만약 지수가 0이면
    if exp == 0:
        # 1 반환
        return 1
    # 만약 지수가 1이면
    elif exp == 1:
        # A 자기 자신 반환
        return base % mod
    # 만약 지수가 짝수면
    elif exp % 2 == 0:
        # 재귀적으로 반으로 나눈 지수 계산
        half = fast_power(base, exp // 2, mod)
        # 제곱하고 mod 연산 적용
        return (half * half) % mod
    # 만약 지수가 홀수면
    else:
        # 재귀적으로 지수-1 거듭제곱 계산 후 base 곱하기
        return (base * fast_power(base, exp - 1, mod)) % mod

def main() -> None:
    A: int
    B: int
    C: int
    A, B, C = map(int, input().split()) # 입력 고정
    
    result: int = fast_power(A, B, C) # 출력 형식
    print(result) 

if __name__ == "__main__":
    main()
