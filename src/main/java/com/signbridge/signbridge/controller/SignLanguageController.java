package com.signbridge.signbridge.controller;

import com.signbridge.signbridge.dto.FrameRequest;
import com.signbridge.signbridge.dto.PredictionResponse;
import com.signbridge.signbridge.service.PythonMLService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.Arrays;

@RestController
@RequestMapping("/sign-language")
public class SignLanguageController {

    private final PythonMLService pythonMLService;

    public SignLanguageController(PythonMLService pythonMLService) {
        this.pythonMLService = pythonMLService;
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "UP");
        response.put("service", "Sign Language Detection API");
        response.put("pythonServiceStatus", pythonMLService.isPythonServiceRunning() ? "UP" : "DOWN");
        return ResponseEntity.ok(response);
    }

    @PostMapping("/predict")
    public ResponseEntity<PredictionResponse> predictSign(@RequestBody FrameRequest frameRequest) {
        if (frameRequest.getFrameData() == null || frameRequest.getFrameData().isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(new PredictionResponse(false, "Frame data is required"));
        }

        if (!pythonMLService.isPythonServiceRunning()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(new PredictionResponse(false, "Python ML service is not running"));
        }

        PredictionResponse prediction = pythonMLService.predictFromFrame(frameRequest.getFrameData());

        if (prediction.isSuccess()) {
            return ResponseEntity.ok(prediction);
        } else {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(prediction);
        }
    }

    @PostMapping("/predict/video")
    public ResponseEntity<PredictionResponse> predictFromVideo(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(new PredictionResponse(false, "Video file is required"));
        }

        PredictionResponse prediction = pythonMLService.predictFromVideo(file);
        return ResponseEntity.ok(prediction);
    }

    @GetMapping("/python-service/status")
    public ResponseEntity<Map<String, Object>> getPythonServiceStatus() {
        Map<String, Object> response = new HashMap<>();
        boolean isRunning = pythonMLService.isPythonServiceRunning();
        response.put("running", isRunning);
        response.put("status", isRunning ? "ACTIVE" : "INACTIVE");
        return ResponseEntity.ok(response);
    }

    @PostMapping("/python-service/start")
    public ResponseEntity<Map<String, String>> startPythonService() {
        String result = pythonMLService.startPythonService();
        Map<String, String> response = new HashMap<>();
        response.put("message", result);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/actions")
    public ResponseEntity<Map<String, String>> getAvailableActions() {
        Map<String, String> actions = new HashMap<>();
        actions.put("A", "Hello");
        actions.put("B", "Yes");
        actions.put("C", "No");
        actions.put("D", "Thank You");
        actions.put("E", "I Love You");
        actions.put("F", "See You Again");
        return ResponseEntity.ok(actions);
    }

    @GetMapping("/dictionary")
    public ResponseEntity<List<String>> getDictionary() {
        try {
            List<String> signs = Arrays.asList(
                    "Hello", "Hi", "Good evening", "How are u",
                    "I am fine", "I need water", "Thank You");
            return ResponseEntity.ok(signs);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}