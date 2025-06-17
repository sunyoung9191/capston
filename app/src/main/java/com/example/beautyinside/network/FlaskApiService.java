package com.example.beautyinside.network;

import java.util.List;
import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import com.example.beautyinside.ReviewItem;

public interface FlaskApiService {

    @Multipart
    @POST("/predict")
    Call<List<ReviewItem>> sendImage(
            @Part MultipartBody.Part image
    );
}
