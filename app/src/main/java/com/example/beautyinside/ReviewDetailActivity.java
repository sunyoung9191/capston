package com.example.beautyinside;

import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.bumptech.glide.Glide;

public class ReviewDetailActivity extends AppCompatActivity {

    private TextView textUserName, textTags, textReview;
    private ImageView imageBefore, imageAfter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_review_detail);

        textUserName = findViewById(R.id.textNickname);
        textTags = findViewById(R.id.textTags);
        textReview = findViewById(R.id.textReview);
        imageBefore = findViewById(R.id.imageBefore);
        imageAfter = findViewById(R.id.imageAfter);

        ReviewData data = (ReviewData) getIntent().getSerializableExtra("reviewData");

        if (data != null) {
            textUserName.setText(data.getUserName());
            textTags.setText("#" + data.getType());
            textReview.setText(data.getText());

            String beforePath = "file:///android_asset/images/" + data.getBeforeImage();
            String afterPath = "file:///android_asset/images/" + data.getAfterImage();

            Glide.with(this).load(beforePath).into(imageBefore);
            Glide.with(this).load(afterPath).into(imageAfter);
        }
    }
}
