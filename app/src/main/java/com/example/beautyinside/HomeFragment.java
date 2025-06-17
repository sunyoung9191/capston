package com.example.beautyinside;
import com.example.beautyinside.network.RetrofitClient;
import com.example.beautyinside.network.FlaskApiService;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;
import android.os.Handler;
import android.os.Looper;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.viewpager2.widget.ViewPager2;
import java.util.ArrayList;
import java.util.List;
import android.widget.Button;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.database.Cursor;
import androidx.annotation.Nullable;

import java.io.File;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import java.io.InputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import android.net.Uri;
import android.content.Intent;
import android.provider.MediaStore;
import android.app.Activity;

import com.example.beautyinside.network.FlaskApiService;
import com.google.gson.JsonObject;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import android.util.Log;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

import java.io.FileOutputStream;
import java.io.IOException;

public class HomeFragment extends Fragment {

    private static final int PICK_IMAGE_REQUEST = 100;
    private boolean useLandmarkServer = true; private static final int REQUEST_CODE_GALLERY = 101;
    private static final int REQUEST_CODE_CAMERA = 102;
    private static final int REQUEST_PERMISSION = 100;
    private ViewPager2 bannerViewPager;
    private Handler sliderHandler = new Handler(Looper.getMainLooper());
    private Runnable sliderRunnable;
    private int currentPage = 0;



    public HomeFragment() {
        // 기본 생성자
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_home, container, false);

        ImageButton btnAddImage = view.findViewById(R.id.btnAddImage);
        EditText searchBar = view.findViewById(R.id.search_bar);
        ImageView searchIcon = view.findViewById(R.id.search_icon);

        super.onViewCreated(view, savedInstanceState);

        // 🔐 Android 13 이상: READ_MEDIA_IMAGES 퍼미션 요청
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(requireActivity(),
                        new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 100);
            }
        }
        bannerViewPager = view.findViewById(R.id.bannerViewPager);

        // 배너 이미지 리스트 생성
        List<Integer> bannerImages = new ArrayList<>();
        bannerImages.add(R.drawable.banner_1);
        bannerImages.add(R.drawable.banner_2);
        bannerImages.add(R.drawable.banner_3);
        bannerImages.add(R.drawable.banner_4);
        bannerImages.add(R.drawable.banner_5);

        // 어댑터 설정
        BannerAdapter adapter = new BannerAdapter(requireContext(), bannerImages,  R.layout.item_banner_image); // 세 번째 인자는 예시로 0 넣었어
        bannerViewPager.setAdapter(adapter);


        // 페이지 바뀔 때 현재 페이지 위치 저장
        bannerViewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                currentPage = position;
            }
        });

        // 자동 슬라이드 설정
        sliderRunnable = new Runnable() {
            @Override
            public void run() {
                if (adapter.getItemCount() == 0) return;
                currentPage = (currentPage + 1) % adapter.getItemCount();
                bannerViewPager.setCurrentItem(currentPage, true);
                sliderHandler.postDelayed(this, 3000); // 3초마다 변경
            }
        };

        sliderHandler.postDelayed(sliderRunnable, 3000);
        btnAddImage.setOnClickListener(v -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED
                        || ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

                    ActivityCompat.requestPermissions(requireActivity(),
                            new String[]{Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.CAMERA},
                            REQUEST_PERMISSION);
                } else {
                    showImagePickerDialog();
                }
            } else {
                if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                        || ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

                    ActivityCompat.requestPermissions(requireActivity(),
                            new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.CAMERA},
                            REQUEST_PERMISSION);
                } else {
                    showImagePickerDialog();
                }
            }
        });


        searchIcon.setOnClickListener(v -> {
            String query = searchBar.getText().toString();
            if (!query.isEmpty()) {
                Intent intent = new Intent(requireContext(), SearchResultActivity.class);
                intent.putExtra("query", query);
                startActivity(intent);
            } else {
                Toast.makeText(requireContext(), "검색어를 입력하세요", Toast.LENGTH_SHORT).show();
            }
        });



        // 로그인/회원가입 버튼 연결
        Button loginSignupButton = view.findViewById(R.id.btnLoginSignup);

        // 클릭 이벤트 연결
        loginSignupButton.setOnClickListener(v -> {
            Intent intent = new Intent(getActivity(), LoginActivity.class);
            startActivity(intent);
        });

        return view;
    }
    private File getFileFromUri(Uri uri) {
        try {
            InputStream inputStream = requireContext().getContentResolver().openInputStream(uri);
            File tempFile = new File(requireContext().getCacheDir(), "selected_image.jpg");
            FileOutputStream outputStream = new FileOutputStream(tempFile);

            byte[] buffer = new byte[1024];
            int len;
            while ((len = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, len);
            }

            outputStream.close();
            inputStream.close();

            return tempFile;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private void showImagePickerDialog() {
        String[] options = {"갤러리에서 선택", "카메라로 촬영"};

        new android.app.AlertDialog.Builder(requireContext())
                .setTitle("이미지 추가")
                .setItems(options, (dialog, which) -> {
                    if (which == 0) {
                        openGallery();
                    } else {
                        openCamera();
                    }
                })
                .show();
    }


    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_CODE_GALLERY);
    }
    public void onDestroyView() {
        super.onDestroyView();
        // 자동 슬라이드 중지
        sliderHandler.removeCallbacks(sliderRunnable);
    }
    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CODE_CAMERA);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                showImagePickerDialog();
            } else {
                Toast.makeText(requireContext(), "권한이 필요합니다", Toast.LENGTH_SHORT).show();
            }
        }
    }



    private String getRealPathFromURI(Uri contentUri) {
        String[] proj = { MediaStore.Images.Media.DATA };
        Cursor cursor = getActivity().getContentResolver().query(contentUri, proj, null, null, null);
        if (cursor == null) return null;
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        String path = cursor.getString(column_index);
        cursor.close();
        return path;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            Log.d("예은디버그", "✅ 선택된 이미지 URI: " + selectedImage);

            File imageFile = getFileFromUri(selectedImage);  // ✅ 여기서 경로 대신 파일 생성
            if (imageFile != null && imageFile.exists()) {
                Log.d("예은디버그", "✅ 파일 생성 완료: " + imageFile.getAbsolutePath());
                sendImageToServer(imageFile);
            } else {
                Log.e("예은디버그", "❌ 이미지 파일 생성 실패");
            }
        } else {
            Log.e("예은디버그", "❌ 사진 선택 실패 or 취소");
        }
    }





    private File saveBitmapToFile(Bitmap bitmap) {
        try {
            File file = new File(requireContext().getCacheDir(), "captured_image.jpg");
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
            return file;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }


    private void sendImageToServer(File imageFile) {
        Log.d("예은디버그", "🔄 서버 전송 시작: " + imageFile.getAbsolutePath());

        RequestBody reqFile = RequestBody.create(MediaType.parse("image/*"), imageFile);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", imageFile.getName(), reqFile);

        // ✅ 하나로 통합된 RetrofitClient 사용
        FlaskApiService service = RetrofitClient.getApiService();

        Call<List<ReviewItem>> call = service.sendImage(body);
        call.enqueue(new Callback<List<ReviewItem>>() {
            @Override
            public void onResponse(Call<List<ReviewItem>> call, Response<List<ReviewItem>> response) {
                Log.d("예은디버그", "✅ 서버 응답 도착! 코드: " + response.code());

                if (response.isSuccessful() && response.body() != null) {
                    List<ReviewItem> reviewList = response.body();
                    Log.d("예은디버그", "✅ 응답 받은 ReviewItem 개수: " + reviewList.size());

                    // ✅ JSON 변환해서 넘기기
                    Gson gson = new Gson();
                    String jsonList = gson.toJson(reviewList);

                    Intent intent = new Intent(requireContext(), SimilarReviewActivity.class);
                    intent.putExtra("reviewList", jsonList);
                    startActivity(intent);

                } else {
                    Log.e("예은디버그", "❌ 응답 실패 - HTTP 코드: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<List<ReviewItem>> call, Throwable t) {
                Log.e("예은디버그", "❌ 서버 연결 실패: " + t.getMessage(), t);
            }
        });
    }

}






