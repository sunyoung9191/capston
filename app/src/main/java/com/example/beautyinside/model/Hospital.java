package com.example.beautyinside.model;

public class Hospital {
    private String name;
    private String description;
    private String imageKey;

    public Hospital(String name, String description, String imageKey) {
        this.name = name;
        this.description = description;
        this.imageKey = imageKey;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getImageKey() {
        return imageKey;
    }
}
