package com.deepfillv2.api.admin;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/** 운영자 콘솔 API. X-Admin-Key로 보호한다. */
@RestController
@RequestMapping("/api/admin")
public class AdminController {

    private final AuditService auditService;
    private final String adminKey;

    public AdminController(AuditService auditService, @Value("${admin.key}") String adminKey) {
        this.auditService = auditService;
        this.adminKey = adminKey;
    }

    @GetMapping("/stats")
    public ResponseEntity<?> stats(@RequestHeader(value = "X-Admin-Key", required = false) String key) {
        if (key == null || !adminKey.equals(key)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("status", "error", "message", "유효한 X-Admin-Key 헤더가 필요합니다."));
        }
        return ResponseEntity.ok(auditService.stats());
    }
}
