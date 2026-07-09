package com.deepfillv2.api.inpaint;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.util.Map;

/**
 * API 공통 에러 응답.
 * 어떤 실패든 프론트가 그대로 보여줄 수 있는 한국어 message를 담은 JSON으로 통일한다.
 */
@RestControllerAdvice
public class ApiExceptionHandler {

    @ExceptionHandler(InvalidUploadException.class)
    public ResponseEntity<Map<String, String>> invalidUpload(InvalidUploadException e) {
        return error(HttpStatus.BAD_REQUEST, e.getMessage());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<Map<String, String>> tooLarge(MaxUploadSizeExceededException e) {
        return error(HttpStatus.PAYLOAD_TOO_LARGE, "이미지가 너무 큽니다. 20MB 이하 파일로 다시 시도해주세요.");
    }

    @ExceptionHandler(ResourceAccessException.class)
    public ResponseEntity<Map<String, String>> inferenceDown(ResourceAccessException e) {
        return error(HttpStatus.SERVICE_UNAVAILABLE, "AI 추론 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.");
    }

    @ExceptionHandler(RestClientResponseException.class)
    public ResponseEntity<Map<String, String>> inferenceError(RestClientResponseException e) {
        return error(HttpStatus.BAD_GATEWAY, "AI 처리 중 오류가 발생했습니다. 다른 이미지로 다시 시도해주세요.");
    }

    private ResponseEntity<Map<String, String>> error(HttpStatus status, String message) {
        return ResponseEntity.status(status).body(Map.of("status", "error", "message", message));
    }
}
