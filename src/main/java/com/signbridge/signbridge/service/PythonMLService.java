package com.signbridge.signbridge.service;

import com.signbridge.signbridge.dto.PredictionResponse;
import org.springframework.web.multipart.MultipartFile;

public interface PythonMLService {
    PredictionResponse predictFromFrame(String base64Frame);
    PredictionResponse predictFromVideo(MultipartFile videoFile);
    boolean isPythonServiceRunning();
    String startPythonService();
}
