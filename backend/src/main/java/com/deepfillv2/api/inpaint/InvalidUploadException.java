package com.deepfillv2.api.inpaint;

/** 업로드 파일이 비어 있거나 이미지가 아닐 때 던지는 예외. */
public class InvalidUploadException extends RuntimeException {

    public InvalidUploadException(String message) {
        super(message);
    }
}
