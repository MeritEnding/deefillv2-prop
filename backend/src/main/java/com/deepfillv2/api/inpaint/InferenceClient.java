package com.deepfillv2.api.inpaint;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

/**
 * Python 추론 서비스(model/serving) 호출 클라이언트.
 */
@Component
public class InferenceClient {

    private final RestClient restClient;

    public InferenceClient(@Value("${inference.base-url}") String baseUrl) {
        this.restClient = RestClient.builder().baseUrl(baseUrl).build();
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

    @SuppressWarnings("unchecked")
    public Map<String, String> health() {
        return restClient.get().uri("/health").retrieve().body(Map.class);
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
