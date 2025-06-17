package com.example.beautyinside;
import java.io.Serializable;

// HospitalData.java
public class HospitalData implements Serializable {
    private String name;
    private int bannerImageResId;

    public HospitalData(String name, int bannerImageResId) {
        this.name = name;
        this.bannerImageResId = bannerImageResId;
    }

    public String getName() {
        return name;
    }

    public int getBannerImageResId() {
        return bannerImageResId;
    }
}
