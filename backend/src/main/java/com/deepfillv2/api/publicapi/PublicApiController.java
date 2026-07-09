package com.deepfillv2.api.publicapi;

import com.deepfillv2.api.admin.AuditService;
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
import java.util.stream.Collectors;

/**
 * 외부 시스템 연동용 공개 API (v1).
 * X-API-Key 인증 후 비식별화를 원콜로 처리하고, 감사 로그·키별 사용량을 남긴다.
 */
@RestController
@RequestMapping("/api/v1")
public class PublicApiController {

    private final InferenceClient inferenceClient;
    private final AuditService auditService;
    private final Set<String> apiKeys;

    public PublicApiController(InferenceClient inferenceClient,
                               AuditService auditService,
                               @Value("${api.keys}") String keys) {
        this.inferenceClient = inferenceClient;
        this.auditService = auditService;
        this.apiKeys = Arrays.stream(keys.split(","))
                .map(String::trim)
                .filter(k -> !k.isEmpty())
                .collect(Collectors.toSet());
    }

    /** 원콜 비식별화: 업로드 → 얼굴·번호판 자동 탐지 → 제거·배경 복원 PNG 반환. */
    @PostMapping("/redact")
    public ResponseEntity<?> redact(@RequestHeader(value = "X-API-Key", required = false) String key,
                                    @RequestPart("image") MultipartFile image) throws IOException {
        if (!authorized(key)) {
            return unauthorized();
        }
        if (image == null || image.isEmpty()) {
            throw new InvalidUploadException("image 파트가 비어 있습니다.");
        }

        long started = System.currentTimeMillis();
        ResponseEntity<byte[]> result = inferenceClient.redact(image);
        long elapsed = System.currentTimeMillis() - started;

        String redactedCount = result.getHeaders().getFirst("X-Redacted-Count");
        int detected = redactedCount != null ? Integer.parseInt(redactedCount) : 0;
        String masked = mask(key);
        auditService.countApiKey(masked);
        auditService.record(masked, "redact", detected, elapsed, "ok");

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header("X-Redacted-Count", String.valueOf(detected))
                .body(result.getBody());
    }

    /** 키별 누적 처리 건수 조회. */
    @GetMapping("/usage")
    public ResponseEntity<?> usage(@RequestHeader(value = "X-API-Key", required = false) String key) {
        if (!authorized(key)) {
            return unauthorized();
        }
        String masked = mask(key);
        return ResponseEntity.ok(Map.of(
                "key", masked,
                "redactCount", String.valueOf(auditService.apiKeyCount(masked))
        ));
    }

    private boolean authorized(String key) {
        return key != null && apiKeys.contains(key);
    }

    private ResponseEntity<Map<String, String>> unauthorized() {
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                .body(Map.of("status", "error", "message", "유효한 X-API-Key 헤더가 필요합니다."));
    }

    private String mask(String key) {
        return key.substring(0, Math.min(4, key.length())) + "****";
    }
}
