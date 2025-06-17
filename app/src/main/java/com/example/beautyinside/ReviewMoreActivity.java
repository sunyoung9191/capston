package com.example.beautyinside;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import android.widget.Toast;
import android.content.Intent;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import java.util.HashMap;
import android.view.View;

public class ReviewMoreActivity extends AppCompatActivity {

    private RecyclerView recyclerView;
    private ReviewAdapter adapter;
    private Spinner doctorSpinner;
    private List<ReviewData> allReviews = new ArrayList<>();
    private static final Set<String> MULTI_DOCTOR_HOSPITALS = new HashSet<>(
            Arrays.asList("나나성형외과", "루호성형외과")
    );

    private Map<String, List<ReviewData>> doctorReviewMap = new HashMap<>();
    private Map<String, Integer> nicknameCounterMap = new HashMap<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_review_more);

        doctorSpinner = findViewById(R.id.doctorSpinner);
        recyclerView = findViewById(R.id.recyclerMoreReviews);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        Intent intent = getIntent();
        String hospitalName = intent.getStringExtra("hospitalName");

        List<ReviewData> defaultReviewList = new ArrayList<>();

        try {
            InputStream inputStream = getAssets().open("data.csv");
            CSVReader reader = new CSVReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));
            String[] line;

            while ((line = reader.readNext()) != null) {
                if (line.length >= 5) {
                    String clinic = line[0].trim();
                    String doctorName = line[1].trim();
                    String procedureType = line[2].trim();
                    String beforeImage = line[3].trim();
                    String afterImage = line[4].trim();

                    String cleanedClinic = clinic.replaceAll("\\s+", "").toLowerCase();
                    String cleanedHospital = hospitalName.replaceAll("\\s+", "").toLowerCase();

                    if (cleanedClinic.equals(cleanedHospital)) {
                        String nicknamePrefix = generateNicknamePrefix(clinic);
                        int count = nicknameCounterMap.getOrDefault(nicknamePrefix, 0) + 1;
                        nicknameCounterMap.put(nicknamePrefix, count);
                        String userName = nicknamePrefix + count;

                        String[] procedures = procedureType.split("#");
                        for (String proc : procedures) {
                            String cleanedProc = proc.trim();
                            if (!cleanedProc.isEmpty()) {
                                ReviewData data = new ReviewData(
                                        userName,
                                        "#" + cleanedProc,
                                        "후기 내용입니다.",
                                        beforeImage,
                                        afterImage,
                                        doctorName
                                );
                                allReviews.add(data);
                                if (MULTI_DOCTOR_HOSPITALS.contains(hospitalName)) {
                                    doctorReviewMap.computeIfAbsent(doctorName, k -> new ArrayList<>()).add(data);
                                } else {
                                    defaultReviewList.add(data);
                                }
                            }
                        }
                    }
                }
            }
            reader.close();

            if (MULTI_DOCTOR_HOSPITALS.contains(hospitalName)) {
                List<String> doctorNames = new ArrayList<>(doctorReviewMap.keySet());
                ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, doctorNames);
                spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                doctorSpinner.setAdapter(spinnerAdapter);

                doctorSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                        String selectedDoctor = doctorNames.get(position);
                        updateRecyclerView(doctorReviewMap.get(selectedDoctor));
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {}
                });
                doctorSpinner.setVisibility(View.VISIBLE);
            } else {
                doctorSpinner.setVisibility(View.GONE);
                updateRecyclerView(defaultReviewList);
            }

        } catch (IOException | CsvValidationException e) {
            e.printStackTrace();
            Log.e("ReviewMoreActivity", "CSV 처리 실패: " + e.getMessage());
        }
    }

    private void updateRecyclerView(List<ReviewData> reviews) {
        adapter = new ReviewAdapter(this, reviews);
        recyclerView.setAdapter(adapter);
    }

    private String generateNicknamePrefix(String hospitalName) {
        if (hospitalName.contains("바바")) return "baba";
        if (hospitalName.contains("메이드영")) return "made";
        if (hospitalName.contains("유노")) return "youno";
        if (hospitalName.contains("옐로우")) return "yellow";
        if (hospitalName.contains("히트")) return "hit";
        if (hospitalName.contains("마블")) return "marble";
        if (hospitalName.contains("온에어")) return "onair";
        if (hospitalName.contains("팝")) return "pop";
        if (hospitalName.contains("티에스")) return "ts";
        if (hospitalName.contains("마인드")) return "mind";
        if (hospitalName.contains("아몬드")) return "amond";
        if (hospitalName.contains("드레스")) return "dress";
        if (hospitalName.contains("디엠")) return "dm";
        if (hospitalName.contains("클래시")) return "clasy";
        if (hospitalName.contains("나나")) return "nana";
        if (hospitalName.contains("루호")) return "ruho";
        return "user";
    }
}