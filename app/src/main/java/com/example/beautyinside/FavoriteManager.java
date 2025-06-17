package com.example.beautyinside;

import java.util.ArrayList;
import java.util.List;

public class FavoriteManager {
    private static FavoriteManager instance;
    private final List<HospitalData> favoriteList;

    private FavoriteManager() {
        favoriteList = new ArrayList<>();
    }

    public static FavoriteManager getInstance() {
        if (instance == null) {
            instance = new FavoriteManager();
        }
        return instance;
    }

    public void addFavorite(HospitalData hospital) {
        if (!isFavorite(hospital)) {
            favoriteList.add(hospital);
        }
    }

    public void removeFavorite(HospitalData hospital) {
        favoriteList.removeIf(item -> item.getName().equals(hospital.getName()));
    }

    public boolean isFavorite(HospitalData hospital) {
        for (HospitalData item : favoriteList) {
            if (item.getName().equals(hospital.getName())) {
                return true;
            }
        }
        return false;
    }

    public List<HospitalData> getFavoriteList() {
        return new ArrayList<>(favoriteList); // 원본 보호
    }
}
