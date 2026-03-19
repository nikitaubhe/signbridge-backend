package com.signbridge.signbridge.service;

import com.signbridge.signbridge.dto.PredictionResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

@Service
public class PythonMLServiceImpl implements PythonMLService {

    private final RestTemplate restTemplate;

    @Value("${python.service.url}")
    private String pythonServiceUrl;

    @Value("${python.script.path}")
    private String pythonScriptPath;

    private Process pythonProcess;

    public PythonMLServiceImpl(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Override
    public PredictionResponse predictFromFrame(String base64Frame) {
        try {
            String url = pythonServiceUrl + "/predict";

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            Map<String, String> requestBody = new HashMap<>();
            requestBody.put("frame", base64Frame);

            HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> body = response.getBody();

                // Safe confidence unwrap — null during warmup frames
                Number confNum = (Number) body.get("confidence");
                Double confidence = confNum != null ? confNum.doubleValue() : null;

                // Pass through all fields from Flask
                PredictionResponse pr = new PredictionResponse(
                        (String) body.get("predictedSign"),
                        (String) body.get("mappedWord"),
                        confidence,
                        (String) body.get("message"),
                        true);

                // Forward warmup/progress fields
                pr.setRequiresMoreFrames(Boolean.TRUE.equals(body.get("requiresMoreFrames")));
                Object prog = body.get("progress");
                Object tot  = body.get("total");
                if (prog != null) pr.setProgress(((Number) prog).intValue());
                if (tot  != null) pr.setTotal(((Number) tot).intValue());

                return pr;
            } else {
                return new PredictionResponse(false, "Failed to get prediction from Python service");
            }

        } catch (Exception e) {
            System.err.println("Error calling Python service: " + e.getMessage());
            return new PredictionResponse(false, "Error: " + e.getMessage());
        }
    }

    @Override
    public PredictionResponse predictFromVideo(MultipartFile videoFile) {
        return new PredictionResponse(false, "Video processing not yet implemented");
    }

    @Override
    public boolean isPythonServiceRunning() {
        try {
            String url = pythonServiceUrl + "/health";
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            return response.getStatusCode() == HttpStatus.OK;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public String startPythonService() {
        try {
            if (isPythonServiceRunning()) {
                return "Python service is already running";
            }

            File scriptDir = new File(pythonScriptPath);
            ProcessBuilder processBuilder = new ProcessBuilder("python", "flask_server.py");
            processBuilder.directory(scriptDir);
            processBuilder.redirectErrorStream(true);

            pythonProcess = processBuilder.start();

            Thread.sleep(3000); // Wait for service to start

            if (isPythonServiceRunning()) {
                return "Python service started successfully";
            } else {
                return "Failed to start Python service";
            }

        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }
}