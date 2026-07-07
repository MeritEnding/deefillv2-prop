package com.deepfillv2.api.inpaint;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

/**
 * 인페인팅 시연 API.
 * 프론트에서 원본 이미지와 마스크(흰색=복원 영역)를 받아
 * 추론 서비스에 전달하고 결과 PNG를 반환한다.
 */
@RestController
@RequestMapping("/api")
public class InpaintController {

    private final InferenceClient inferenceClient;

    public InpaintController(InferenceClient inferenceClient) {
        this.inferenceClient = inferenceClient;
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        try {
            return ResponseEntity.ok(inferenceClient.health());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("status", "down", "detail", "추론 서비스에 연결할 수 없습니다."));
        }
    }

    @PostMapping(value = "/inpaint", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<byte[]> inpaint(@RequestPart("image") MultipartFile image,
                                          @RequestPart("mask") MultipartFile mask) throws IOException {
        if (image.isEmpty() || mask.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        byte[] result = inferenceClient.inpaint(image, mask);
        return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(result);
    }
}
