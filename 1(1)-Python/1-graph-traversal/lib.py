from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        # 인접 리스트를 초기화
        self.graph : DefaultDict[int, list[int]] = defaultdict(list)

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        """
        # 정점 u의 인접 리스트에 v를 추가
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현
        """
        # 방문 여부 저장 리스트 만들기
        visited = [False] * (self.n + 1)
        # 결과 저장 리스트 만들기
        result = []
        # 스택 초기화
        stack = [start]

        # 스택이 빌 때까지
        while stack:
            # 스택에서 정점 꺼내기
            node = stack.pop()
            # 방문한 적 없으면
            if not visited[node]:
                # 방문했다고 표시
                visited[node] = True
                # 결과에 추가
                result.append(node)

                # 인접 정점들에 대해
                for neighbor in sorted(self.graph[node], reverse=True): #역순으로 정렬
                    # 방문한 적 없으면 스택에 추가
                    if not visited[neighbor]:
                        stack.append(neighbor)

        #결과 반환
        return result
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현
        """
        # 방문 여부 저장 리스트 만들기
        visited = [False] * (self.n + 1)
        # 결과 저장 리스트 만들기
        result = []
        # 큐 초기화
        queue = deque([start])
        visited[start] = True

        # 큐가 빌 때까지
        while queue:
            # 큐에서 정점 꺼내기
            node = queue.popleft()
            # 결과에 추가
            result.append(node)

            # 인접 정점들에 대해
            for neighbor in sorted(self.graph[node]):
                # 방문한적 없으면
                if not visited[neighbor]:
                    # 방문했다고 표시
                    visited[neighbor] = True
                    # 큐에 추가
                    queue.append(neighbor)

        # 결과 반환
        return result
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
