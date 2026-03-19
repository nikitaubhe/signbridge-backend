package com.signbridge.signbridge.dto;

public class PredictionResponse {
    private String predictedSign;
    private String mappedWord;
    private Double confidence;
    private String message;
    private boolean success;
    private boolean requiresMoreFrames;
    private Integer progress;
    private Integer total;

    public PredictionResponse() {}

    public PredictionResponse(String predictedSign, String mappedWord, Double confidence,
                              String message, boolean success) {
        this.predictedSign = predictedSign;
        this.mappedWord    = mappedWord;
        this.confidence    = confidence;
        this.message       = message;
        this.success       = success;
    }

    public PredictionResponse(boolean success, String message) {
        this.success = success;
        this.message = message;
    }

    public String getPredictedSign()                    { return predictedSign; }
    public void   setPredictedSign(String v)            { this.predictedSign = v; }

    public String getMappedWord()                       { return mappedWord; }
    public void   setMappedWord(String v)               { this.mappedWord = v; }

    public Double getConfidence()                       { return confidence; }
    public void   setConfidence(Double v)               { this.confidence = v; }

    public String getMessage()                          { return message; }
    public void   setMessage(String v)                  { this.message = v; }

    public boolean isSuccess()                          { return success; }
    public void    setSuccess(boolean v)                { this.success = v; }

    public boolean isRequiresMoreFrames()               { return requiresMoreFrames; }
    public void    setRequiresMoreFrames(boolean v)     { this.requiresMoreFrames = v; }

    public Integer getProgress()                        { return progress; }
    public void    setProgress(Integer v)               { this.progress = v; }

    public Integer getTotal()                           { return total; }
    public void    setTotal(Integer v)                  { this.total = v; }
}