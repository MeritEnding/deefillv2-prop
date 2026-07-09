package com.deepfillv2.api.publicapi;

import com.deepfillv2.api.inpaint.InferenceClient;
import com.deepfillv2.api.inpaint.InvalidUploadException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * 외부 시스템 연동용 공개 API (v1).
 * X-API-Key 인증 후 비식별화를 원콜로 처리하고, 키별 사용량을 집계한다.
 */
@RestController
@RequestMapping("/api/v1")
public class PublicApiController {

    private final InferenceClient inferenceClient;
    private final Set<String> apiKeys;
    private final Map<String, AtomicLong> usage = new ConcurrentHashMap<>();

    public PublicApiController(InferenceClient inferenceClient,
                               @Value("${api.keys}") String keys) {
        this.inferenceClient = inferenceClient;
        this.apiKeys = Arrays.stream(keys.split(","))
                .map(String::trim)
                .filter(k -> !k.isEmpty())
                .collect(Collectors.toSet());
    }

    /** 원콜 비식별화: 업로드 → 얼굴 자동 탐지 → 제거·배경 복원 PNG 반환. */
    @PostMapping("/redact")
    public ResponseEntity<?> redact(@RequestHeader(value = "X-API-Key", required = false) String key,
                                    @RequestPart("image") MultipartFile image) throws IOException {
        ResponseEntity<Map<String, String>> denied = authorize(key);
        if (denied != null) {
            return denied;
        }
        if (image == null || image.isEmpty()) {
            throw new InvalidUploadException("image 파트가 비어 있습니다.");
        }

        ResponseEntity<byte[]> result = inferenceClient.redact(image);
        usage.get(key).incrementAndGet();

        String redactedCount = result.getHeaders().getFirst("X-Redacted-Count");
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header("X-Redacted-Count", redactedCount != null ? redactedCount : "0")
                .body(result.getBody());
    }

    /** 키별 누적 처리 건수 조회. */
    @GetMapping("/usage")
    public ResponseEntity<?> usage(@RequestHeader(value = "X-API-Key", required = false) String key) {
        ResponseEntity<Map<String, String>> denied = authorize(key);
        if (denied != null) {
            return denied;
        }
        long count = usage.get(key).get();
        return ResponseEntity.ok(Map.of(
                "key", key.substring(0, Math.min(4, key.length())) + "****",
                "redactCount", String.valueOf(count)
        ));
    }

    /** 키가 유효하면 usage 슬롯을 준비하고 null, 아니면 401 응답을 반환한다. */
    private ResponseEntity<Map<String, String>> authorize(String key) {
        if (key == null || !apiKeys.contains(key)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("status", "error", "message", "유효한 X-API-Key 헤더가 필요합니다."));
        }
        usage.computeIfAbsent(key, k -> new AtomicLong());
        return null;
    }
}
