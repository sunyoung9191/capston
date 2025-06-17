package com.example.beautyinside;


import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.List;

public class SimilarReviewActivity extends AppCompatActivity {

    private RecyclerView recyclerView;
    private SimilarReviewAdapter adapter;
    private List<ReviewItem> reviewList;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_similar_reviews);

        recyclerView = findViewById(R.id.recycler_similar_reviews);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // 로그로 진입 확인
        Log.d("예은디버그", "✅ SimilarReviewActivity onCreate 진입!");

        // 리뷰 리스트 받아오기
        String jsonList = getIntent().getStringExtra("reviewList");
        Log.d("예은디버그", "✅ 전달받은 JSON: " + jsonList);

        if (jsonList != null) {
            try {
                Type listType = new TypeToken<List<ReviewItem>>() {}.getType();
                reviewList = new Gson().fromJson(jsonList, listType);
                Log.d("예은디버그", "✅ 파싱된 ReviewItem 개수: " + reviewList.size());

                SimilarReviewAdapter adapter = new SimilarReviewAdapter(SimilarReviewActivity.this, reviewList);
                recyclerView.setAdapter(adapter);

            } catch (Exception e) {
                Log.e("예은디버그", "❌ JSON 파싱 또는 어댑터 오류", e);
            }
        } else {
            Log.e("예은디버그", "❌ reviewList 인텐트로 안 넘어옴");
        }
    }


}
