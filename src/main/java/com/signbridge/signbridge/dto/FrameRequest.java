package com.signbridge.signbridge.dto;

public class FrameRequest {
    private String frameData; // Base64 encoded image
    private Integer sequenceNumber;

    public FrameRequest() {
    }

    public FrameRequest(String frameData, Integer sequenceNumber) {
        this.frameData = frameData;
        this.sequenceNumber = sequenceNumber;
    }

    public String getFrameData() {
        return frameData;
    }

    public void setFrameData(String frameData) {
        this.frameData = frameData;
    }

    public Integer getSequenceNumber() {
        return sequenceNumber;
    }

    public void setSequenceNumber(Integer sequenceNumber) {
        this.sequenceNumber = sequenceNumber;
    }
}