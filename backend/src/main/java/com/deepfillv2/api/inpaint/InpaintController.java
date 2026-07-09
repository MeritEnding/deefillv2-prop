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
 * 인페인팅 API.
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
        validateImage(image, "원본 이미지");
        validateImage(mask, "마스크");

        long started = System.currentTimeMillis();
        byte[] result = inferenceClient.inpaint(image, mask);
        long elapsed = System.currentTimeMillis() - started;

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header("Content-Disposition", "inline; filename=\"jium-result.png\"")
                .header("X-Engine", inferenceClient.engineName())
                .header("X-Elapsed-Ms", String.valueOf(elapsed))
                .body(result);
    }

    private void validateImage(MultipartFile file, String label) {
        if (file == null || file.isEmpty()) {
            throw new InvalidUploadException(label + " 파일이 비어 있습니다.");
        }
        String type = file.getContentType();
        if (type == null || !type.startsWith("image/")) {
            throw new InvalidUploadException(label + "는 이미지 파일(JPG·PNG)만 업로드할 수 있습니다.");
        }
    }
}
