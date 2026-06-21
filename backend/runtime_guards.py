"""
runtime_guards.py
역할: API rate limit, 사용자별 상태 잠금, FIFO 추론 큐 제공
입력: 요청 식별 키, 제한 규칙, 잠금 키
출력: 재시도 대기 시간 또는 직렬화된 실행 구간
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock, local
from time import monotonic
from typing import Iterator


@dataclass(frozen=True)
class RateLimitRule:
    """
    역할: sliding-window 요청 제한 규칙 표현
    입력: 허용 요청 수, 윈도우 초
    출력: 불변 rate limit 설정
    """

    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        """
        역할: rate limit 규칙 값 검증
        입력: 생성 시 전달된 필드
        출력: 없음
        """
        if self.max_requests <= 0 or self.window_seconds <= 0:
            raise ValueError("rate limit 값은 1 이상이어야 합니다.")


class SlidingWindowRateLimiter:
    """
    역할: 프로세스 내 요청 시각을 보관해 키별 sliding-window 제한 적용
    입력: scope와 요청 식별자를 결합한 키
    출력: 허용 시 0, 제한 시 재시도 대기 초
    """

    def __init__(self) -> None:
        """
        역할: 요청 기록 저장소와 동기화 잠금 초기화
        입력: 없음
        출력: 없음
        """
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def consume(
        self,
        key: str,
        rule: RateLimitRule,
        now: float | None = None,
    ) -> int:
        """
        역할: 요청 1건을 기록하고 허용 여부 계산
        입력: 제한 키, 제한 규칙, 테스트용 단조 시각
        출력: 허용 시 0, 제한 시 올림한 재시도 대기 초
        """
        current = monotonic() if now is None else float(now)
        cutoff = current - rule.window_seconds

        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= rule.max_requests:
                retry_after = max(1, int(events[0] + rule.window_seconds - current) + 1)
                return retry_after

            events.append(current)
            return 0

    def reset(self) -> None:
        """
        역할: 모든 요청 기록 제거
        입력: 없음
        출력: 없음
        """
        with self._lock:
            self._events.clear()


class KeyedLockPool:
    """
    역할: 사용자나 계정 키별 공유 Lock을 안전하게 생성
    입력: hash 가능한 문자열/정수 키
    출력: 해당 키 전용 Lock
    """

    def __init__(self) -> None:
        """
        역할: 키별 잠금 저장소와 저장소 보호 잠금 초기화
        입력: 없음
        출력: 없음
        """
        self._locks: dict[object, Lock] = {}
        self._registry_lock = Lock()

    def get(self, key: object) -> Lock:
        """
        역할: 지정 키의 공유 잠금 반환
        입력: 잠금 식별 키
        출력: 키 전용 threading.Lock
        """
        with self._registry_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = Lock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def hold(self, key: object) -> Iterator[None]:
        """
        역할: 지정 키 잠금을 획득한 실행 구간 제공
        입력: 잠금 식별 키
        출력: context manager 실행 구간
        """
        lock = self.get(key)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


class SerializedInferenceQueue:
    """
    역할: 모델 추론 작업을 티켓 순서대로 한 건씩 실행
    입력: slot context에 진입하는 추론 작업
    출력: 동시에 하나만 활성화되는 FIFO 실행 구간
    """

    def __init__(self) -> None:
        """
        역할: FIFO 티켓과 재진입 상태 초기화
        입력: 없음
        출력: 없음
        """
        self._condition = Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._local = local()

    @contextmanager
    def slot(self) -> Iterator[None]:
        """
        역할: 현재 스레드에 FIFO 추론 실행 권한 부여
        입력: 없음
        출력: 직렬화된 context manager 실행 구간
        """
        depth = int(getattr(self._local, "depth", 0))
        if depth > 0:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return

        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket:
                self._condition.wait()

        self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._condition:
                self._serving_ticket += 1
                self._condition.notify_all()

    def queued_count(self) -> int:
        """
        역할: 실행 대기 또는 실행 중인 추론 작업 수 반환
        입력: 없음
        출력: 현재 미처리 티켓 수
        """
        with self._condition:
            return self._next_ticket - self._serving_ticket
