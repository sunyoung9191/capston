package com.example.beautyinside;

import java.util.List;

public class DoctorData {
    private final String name;
    private final int imageResId;
    private final List<String> tags;

    public DoctorData(String name, int imageResId, List<String> tags) {
        this.name = name;
        this.imageResId = imageResId;
        this.tags = tags;
    }

    public String getName() {
        return name;
    }

    public int getImageResId() {
        return imageResId;
    }

    public List<String> getTags() {
        return tags;
    }
}
