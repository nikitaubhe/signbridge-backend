package com.signbridge.signbridge.dto;

public class PredictionResponse {
    private String predictedSign;
    private String mappedWord;
    private Double confidence;
    private String message;
    private boolean success;

    public PredictionResponse() {
    }

    public PredictionResponse(String predictedSign, String mappedWord, Double confidence, String message,
            boolean success) {
        this.predictedSign = predictedSign;
        this.mappedWord = mappedWord;
        this.confidence = confidence;
        this.message = message;
        this.success = success;
    }

    public PredictionResponse(boolean success, String message) {
        this.success = success;
        this.message = message;
    }

    public String getPredictedSign() {
        return predictedSign;
    }

    public void setPredictedSign(String predictedSign) {
        this.predictedSign = predictedSign;
    }

    public String getMappedWord() {
        return mappedWord;
    }

    public void setMappedWord(String mappedWord) {
        this.mappedWord = mappedWord;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }
}