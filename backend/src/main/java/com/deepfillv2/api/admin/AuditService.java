package com.deepfillv2.api.admin;

import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 처리 감사 로그. 최근 이력은 링버퍼(메모리)로, 총계·키별 집계는 카운터로 유지한다.
 * 재시작 시 초기화되는 데모 수준 — 운영 전환 시 DB/로그 수집기로 교체 지점.
 */
@Component
public class AuditService {

    /** 감사 로그 한 건. source는 "web" 또는 마스킹된 API 키. */
    public record Entry(Instant at, String source, String action, int detected, long elapsedMs, String status) {}

    private static final int MAX_ENTRIES = 200;

    private final ConcurrentLinkedDeque<Entry> entries = new ConcurrentLinkedDeque<>();
    private final Map<String, AtomicLong> byAction = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> byApiKey = new ConcurrentHashMap<>();
    private final AtomicLong total = new AtomicLong();
    private final AtomicLong totalElapsedMs = new AtomicLong();

    public void record(String source, String action, int detected, long elapsedMs, String status) {
        entries.addFirst(new Entry(Instant.now(), source, action, detected, elapsedMs, status));
        while (entries.size() > MAX_ENTRIES) {
            entries.pollLast();
        }
        total.incrementAndGet();
        totalElapsedMs.addAndGet(elapsedMs);
        byAction.computeIfAbsent(action, k -> new AtomicLong()).incrementAndGet();
    }

    /** 공개용 요약 — 개별 이력·키 정보 없이 총계만. */
    public Map<String, Object> publicStats() {
        long t = total.get();
        return Map.of(
                "total", t,
                "byAction", snapshot(byAction),
                "avgElapsedMs", t == 0 ? 0 : totalElapsedMs.get() / t
        );
    }

    public long countApiKey(String maskedKey) {
        return byApiKey.computeIfAbsent(maskedKey, k -> new AtomicLong()).incrementAndGet();
    }

    public long apiKeyCount(String maskedKey) {
        AtomicLong c = byApiKey.get(maskedKey);
        return c == null ? 0 : c.get();
    }

    public Map<String, Object> stats() {
        List<Entry> recent = new ArrayList<>(entries);
        return Map.of(
                "total", total.get(),
                "byAction", snapshot(byAction),
                "byApiKey", snapshot(byApiKey),
                "recent", recent
        );
    }

    private Map<String, Long> snapshot(Map<String, AtomicLong> src) {
        Map<String, Long> out = new ConcurrentHashMap<>();
        src.forEach((k, v) -> out.put(k, v.get()));
        return out;
    }
}
