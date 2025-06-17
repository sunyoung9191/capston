package com.example.beautyinside;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.database.Cursor;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.beautyinside.network.FlaskApiService;
import com.example.beautyinside.network.RetrofitClient;
import com.google.gson.Gson;

import java.io.File;
import java.lang.reflect.Type;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import com.google.gson.reflect.TypeToken;

public class PredictActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_SELECT_IMAGE = 101;
    private String imagePath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_similar_reviews);



    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_CODE_SELECT_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_CODE_SELECT_IMAGE && resultCode == Activity.RESULT_OK && data != null) {
            Uri selectedImageUri = data.getData();
            imagePath = getRealPathFromURI(selectedImageUri);
            if (imagePath != null) {
                sendImageToServer(imagePath);
            } else {
                Toast.makeText(this, "이미지 경로를 불러올 수 없습니다", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private String getRealPathFromURI(Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor != null) {
            int idx = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            String result = cursor.getString(idx);
            cursor.close();
            return result;
        }
        return null;
    }

    private void sendImageToServer(String imagePath) {
        File file = new File(imagePath);
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", file.getName(), requestFile);

        FlaskApiService api = RetrofitClient.getApiService();
        Call<List<ReviewItem>> call = api.sendImage(body);

        call.enqueue(new Callback<List<ReviewItem>>() {
            @Override
            public void onResponse(Call<List<ReviewItem>> call, Response<List<ReviewItem>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    List<ReviewItem> resultList = response.body();

                    // JSON으로 직렬화해서 다음 액티비티로 넘기기
                    String jsonList = new Gson().toJson(resultList);

                    Intent intent = new Intent(PredictActivity.this, SimilarReviewActivity.class);
                    intent.putExtra("reviewList", jsonList);
                    startActivity(intent);
                } else {
                    Toast.makeText(PredictActivity.this, "서버 응답 실패", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<List<ReviewItem>> call, Throwable t) {
                Toast.makeText(PredictActivity.this, "서버 통신 오류: " + t.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
    }
}
