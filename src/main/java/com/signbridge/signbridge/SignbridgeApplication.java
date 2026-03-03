package com.signbridge.signbridge;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;

@SpringBootApplication
public class SignbridgeApplication {

	public static void main(String[] args) {

            SpringApplication.run(SignbridgeApplication.class, args);
            System.out.println("===========================================");
            System.out.println("Sign Language Detection API is running!");
            System.out.println("API Base URL: http://localhost:8080/api");
            System.out.println("===========================================");
        }

        @Bean
        public RestTemplate restTemplate() {
            return new RestTemplate();

	}

    @Bean
    public com.fasterxml.jackson.databind.ObjectMapper objectMapper() {
        return new com.fasterxml.jackson.databind.ObjectMapper();
    }

}
