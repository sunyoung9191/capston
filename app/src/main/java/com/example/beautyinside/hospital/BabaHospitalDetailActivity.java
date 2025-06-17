package com.example.beautyinside.hospital;



import android.content.Intent;
import android.widget.Button;
import android.os.Bundle;
import android.os.Handler;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.widget.ViewPager2;
import com.example.beautyinside.R;
import com.example.beautyinside.BannerAdapter;
import com.example.beautyinside.DoctorListAdapter;
import com.example.beautyinside.ReviewAdapter;
import com.example.beautyinside.DoctorData;
import com.example.beautyinside.ReviewData;
import com.example.beautyinside.ReviewMoreActivity;
import java.util.Arrays;
import java.util.List;
import android.net.Uri;
import android.widget.Toast;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.widget.ImageView;
import android.util.Log;
import com.example.beautyinside.HospitalData;
import com.example.beautyinside.FavoriteManager;

public class BabaHospitalDetailActivity extends AppCompatActivity {

    private ViewPager2 bannerViewPager;
    private BannerAdapter bannerAdapter;
    private Handler handler = new Handler();
    private int currentPage = 0;
    private Runnable bannerRunnable;
    private List<Integer> imageList;
    private ImageButton buttonFavorite;
    private RecyclerView reviewRecyclerView;
    private RecyclerView recyclerDoctors;
    private TextView textHospitalName, textRating;
    private ReviewAdapter reviewAdapter;
    private DoctorListAdapter doctorAdapter;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_baba_hospital_detail);

        // 1. 뒤로가기 버튼 연결
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> onBackPressed());

        // 3. 병원 이름, 별점, 찜 버튼 설정
        TextView textHospitalName = findViewById(R.id.textHospitalName);
        TextView textRating = findViewById(R.id.textRating);
        ImageButton buttonFavorite = findViewById(R.id.buttonFavorite);

        textHospitalName.setText("바바 성형외과");
        textRating.setText("★ 9.7");

        HospitalData babaHospital = new HospitalData("바바성형외과", R.drawable.baba_1);

        final boolean[] isFavorite = {FavoriteManager.getInstance().isFavorite(babaHospital)};
        buttonFavorite.setImageResource(isFavorite[0] ? R.drawable.ic_heart_filled : R.drawable.ic_heart_border);

        buttonFavorite.setOnClickListener(v -> {
            isFavorite[0] = !isFavorite[0];

            if (isFavorite[0]) {
                buttonFavorite.setImageResource(R.drawable.ic_heart_filled);
                FavoriteManager.getInstance().addFavorite(babaHospital);
            } else {
                buttonFavorite.setImageResource(R.drawable.ic_heart_border);
                FavoriteManager.getInstance().removeFavorite(babaHospital);
            }
        });

        // 의료진 리스트
        recyclerDoctors = findViewById(R.id.recyclerDoctors);
        recyclerDoctors.setLayoutManager(new LinearLayoutManager(this));
        List<DoctorData> doctors = Arrays.asList(
                new DoctorData(
                        "김상일",
                        R.drawable.doctor_baba_park,
                        Arrays.asList("눈성형", "리프팅")
                )
        );
        DoctorListAdapter doctorAdapter = new DoctorListAdapter(this, doctors);
        recyclerDoctors.setAdapter(doctorAdapter);

        TextView textAddress = findViewById(R.id.textAddress);
        Button buttonCopyAddress = findViewById(R.id.buttonCopyAddress);
        ImageView imageMapPreview = findViewById(R.id.imageMapPreview);

// 주소 복사
        buttonCopyAddress.setOnClickListener(v -> {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            ClipData clip = ClipData.newPlainText("주소", textAddress.getText().toString());
            clipboard.setPrimaryClip(clip);
            Toast.makeText(this, "주소가 복사되었습니다", Toast.LENGTH_SHORT).show();
        });

// 지도 이미지 클릭 시 구글맵 이동
        imageMapPreview.setOnClickListener(v -> {
            Uri gmmIntentUri = Uri.parse("geo:0,0?q=서울 강남구 논현로 857 (신사동, A857) 3층 바바성형외과의원\n");
            Intent mapIntent = new Intent(Intent.ACTION_VIEW, gmmIntentUri);
            mapIntent.setPackage("com.google.android.apps.maps");

            if (mapIntent.resolveActivity(getPackageManager()) != null) {
                startActivity(mapIntent);
            }
        });

        // 2. 배너 이미지 리스트 준비
        List<Integer> imageResIds = Arrays.asList(
                R.drawable.baba_1, R.drawable.baba_2

        );

        //후기 버튼


        Button showMore = findViewById(R.id.buttonShowMore);
        showMore.setOnClickListener(v -> {
            Log.d("FINAL_DEBUG", "🔥 버튼 클릭됨 → startActivity 직전");

            Intent intent = new Intent(BabaHospitalDetailActivity.this, ReviewMoreActivity.class);
            intent.putExtra("hospitalName", "바바성형외과");
            startActivity(intent);
        });




        // 3. ViewPager + 어댑터 연결
        bannerViewPager = findViewById(R.id.bannerViewPager);
        BannerAdapter adapter = new BannerAdapter(this, imageResIds, R.layout.item_banner); // ← 2개 인자 버전 사용 중
        bannerViewPager.setAdapter(adapter);

        // 4. 자동 슬라이드 (2초 간격)
        bannerRunnable = new Runnable() {
            @Override
            public void run() {
                currentPage = (currentPage + 1) % imageResIds.size();
                bannerViewPager.setCurrentItem(currentPage, true);
                handler.postDelayed(this, 2000); // ✅ 2초 간격
            }
        };
        handler.postDelayed(bannerRunnable, 2000);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacks(bannerRunnable);
    }
}