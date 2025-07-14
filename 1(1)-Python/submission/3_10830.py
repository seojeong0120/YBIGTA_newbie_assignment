from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # 특정 위치의 값 설정하기
        self.matrix[key[0]][key[1]] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        # 정방행렬 검사
        assert self.shape[0] == self.shape[1]

        # n이 0이면
        if n == 0:
            # 항등행렬 반환
            return Matrix.eye(self.shape[0])
        # n이 1이면
        elif n == 1:
            # 자기 자신 반환
            return self.clone()
        # n이 짝수면
        elif n % 2 == 0:
            # 지수를 절반으로 줄여서 제곱
            half = self ** (n // 2)
            # 절반으로 줄인 것에 대해 __matmul__ 수행
            return half @ half
        # n이 홀수면
        else:
            # 재귀적으로 __matmul__ 수행
            return self @ (self ** (n - 1))

    def __repr__(self) -> str:
        # 행들을 담을 리스트 생성
        rows = []
        # 행렬의 각 행마다
        for row in self.matrix:
            # 각 원소들을 문자열로 바꾸기
            elements = " ".join(str(x % self.MOD) for x in row)
            # 행들을 담을 리스트에 추가
            rows.append(elements)

        # 각 줄을 개행 문자로 이어붙이고 최종 문자열 생성
        result = "\n".join(rows)
        #최종 문자열 반환
        return result


from typing import Callable
import sys


"""
-아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()