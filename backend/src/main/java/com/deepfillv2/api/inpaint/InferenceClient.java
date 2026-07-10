package com.deepfillv2.api.inpaint;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.Map;

/**
 * Python 추론 서비스(model/serving) 호출 클라이언트.
 */
@Component
public class InferenceClient {

    private final RestClient restClient;
    private volatile String engineName = "unknown";

    public InferenceClient(@Value("${inference.base-url}") String baseUrl) {
        // uvicorn은 h2c 업그레이드 요청의 본문을 처리하지 못하므로 HTTP/1.1로 고정한다.
        HttpClient httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(3))
                .build();
        this.restClient = RestClient.builder()
                .requestFactory(new JdkClientHttpRequestFactory(httpClient))
                .baseUrl(baseUrl)
                .build();
    }

    public byte[] inpaint(MultipartFile image, MultipartFile mask) throws IOException {
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("image", asResource(image, "image.png"));
        parts.add("mask", asResource(mask, "mask.png"));

        return restClient.post()
                .uri("/inpaint")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(byte[].class);
    }

    public org.springframework.http.ResponseEntity<byte[]> redact(MultipartFile image) throws IOException {
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("image", asResource(image, "image.png"));

        return restClient.post()
                .uri("/redact")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .toEntity(byte[].class);
    }

    public String detect(MultipartFile image, String targets) throws IOException {
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("image", asResource(image, "image.png"));
        parts.add("targets", targets);

        return restClient.post()
                .uri("/detect")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(String.class);
    }

    public byte[] segment(MultipartFile image, double x, double y) throws IOException {
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("image", asResource(image, "image.png"));
        parts.add("x", String.valueOf(x));
        parts.add("y", String.valueOf(y));

        return restClient.post()
                .uri("/segment")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(parts)
                .retrieve()
                .body(byte[].class);
    }

    @SuppressWarnings("unchecked")
    public Map<String, String> health() {
        Map<String, String> body = restClient.get().uri("/health").retrieve().body(Map.class);
        if (body != null && body.get("engine") != null) {
            engineName = body.get("engine");
        }
        return body;
    }

    /** 마지막 health 조회에서 확인한 엔진 이름. */
    public String engineName() {
        return engineName;
    }

    private ByteArrayResource asResource(MultipartFile file, String fallbackName) throws IOException {
        byte[] bytes = file.getBytes();
        String name = file.getOriginalFilename() != null ? file.getOriginalFilename() : fallbackName;
        return new ByteArrayResource(bytes) {
            @Override
            public String getFilename() {
                return name;
            }
        };
    }
}
